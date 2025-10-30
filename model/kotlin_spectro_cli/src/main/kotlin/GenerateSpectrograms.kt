import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import javax.sound.sampled.AudioSystem
import javax.sound.sampled.AudioFormat
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import org.jtransforms.fft.FloatFFT_1D

fun main(args: Array<String>) {
    // Usage: kotlin-spectro-cli <voice_dataset_dir> <output_dir>
    val baseDir = if (args.isNotEmpty()) File(args[0]) else File("../voice_dataset")
    val outDir = if (args.size >= 2) File(args[1]) else File("../spectrograms_from_kotlin")
    require(baseDir.exists() && baseDir.isDirectory) { "voice_dataset not found: ${baseDir.absolutePath}" }
    outDir.mkdirs()

    val labelFolders = listOf(
        "PD_AH" to "PD",
        "HC_AH" to "HC"
    )
    val counters = mutableMapOf("PD" to 0, "HC" to 0)

    for ((sub, label) in labelFolders) {
        val folder = File(baseDir, sub)
        if (!folder.isDirectory) {
            println("Skipping missing folder: ${folder.absolutePath}")
            continue
        }
        println("Generating spectrograms for $label from ${folder.absolutePath}")
        val files = folder.listFiles()?.filter { it.isFile && it.extension.lowercase() == "wav" }?.sortedBy { it.name.lowercase() } ?: emptyList()
        for (f in files) {
            try {
                counters[label] = counters.getValue(label) + 1
                val idx = counters.getValue(label)
                val outName = "%s_%04d.png".format(label, idx)
                val outPng = File(outDir, outName)
                processOne(f, outPng)
                println("Saved: ${outPng.absolutePath}")
            } catch (e: Exception) {
                System.err.println("Error processing ${f.absolutePath}: ${e.message}")
                e.printStackTrace()
            }
        }
    }
    println("Done. PD=${counters["PD"] ?: 0}, HC=${counters["HC"] ?: 0}")
}

// Process one audio file - matches MainActivity.kt exactly
private fun processOne(inFile: File, outPng: File) {
    // Read WAV as mono 22050 Hz (matches readWavPcmAsMono16k from MainActivity.kt)
    // Try fast-path PCM first, fallback to AudioSystem for compressed formats
    val pcm = try {
        readWavPcmAsMono22050(inFile.absolutePath)
    } catch (e: Exception) {
        // Fallback to AudioSystem for non-PCM or compressed WAV files
        decodeWavWithAudioSystem(inFile.absolutePath)
    }
    val sr = 22050
    
    // Compute mel spectrogram (matches computeMelSpectrogram from MainActivity.kt)
    val mel = computeMelSpectrogram(pcm, sr, nMels = 128, nFft = 2048, hop = 512)
    
    // Resize to 299x299 (matches resizeBilinear from MainActivity.kt)
    val targetH = 299
    val targetW = 299
    val resized = resizeBilinear(mel, targetH, targetW)
    
    // Min-max normalization (matches melToBitmap from MainActivity.kt)
    var mn = Float.POSITIVE_INFINITY
    var mx = Float.NEGATIVE_INFINITY
    for (i in 0 until targetH) {
        for (j in 0 until targetW) {
            val v = resized[i][j]
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
    }
    val range = (mx - mn).let { if (it < 1e-6f) 1f else it }
    
    // Create color image with viridis-like colormap (matches Flutter app display)
    // Also flip Y-axis: low frequencies (low mel indices) at bottom, high at top
    val img = BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB)
    for (y in 0 until targetH) {
        // Flip Y-axis: array index 0 (low freq) should be at bottom of image
        val flippedY = targetH - 1 - y
        for (x in 0 until targetW) {
            val norm = ((resized[flippedY][x] - mn) / range).coerceIn(0f, 1f)
            val color = colormapViridis(norm)
            img.setRGB(x, y, color)
        }
    }
    
    outPng.parentFile?.mkdirs()
    ImageIO.write(img, "png", outPng)
}

// Reads WAV (PCM 16-bit) and converts to mono 22050 Hz - matches MainActivity.kt exactly
private fun readWavPcmAsMono22050(path: String): FloatArray {
    RandomAccessFile(File(path), "r").use { raf ->
        fun readLEInt(): Int {
            val b = ByteArray(4)
            raf.readFully(b)
            return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
        }
        fun readLEShort(): Int {
            val b = ByteArray(2)
            raf.readFully(b)
            return (ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF)
        }

        // RIFF header (12 bytes)
        val riff = ByteArray(4); raf.readFully(riff)
        require(String(riff) == "RIFF") { "Not a RIFF file" }
        /* file size */ readLEInt()
        val wave = ByteArray(4); raf.readFully(wave)
        require(String(wave) == "WAVE") { "Not a WAVE file" }

        var sampleRate = -1
        var numChannels = -1
        var bitsPerSample = -1
        var audioFormat = -1
        var dataBytes: ByteArray? = null

        // Iterate chunks
        while (raf.filePointer + 8 <= raf.length()) {
            val idBytes = ByteArray(4); raf.readFully(idBytes)
            val size = readLEInt()
            val id = String(idBytes)
            when (id) {
                "fmt " -> {
                    // Read at least first 16 bytes
                    audioFormat = readLEShort()
                    numChannels = readLEShort()
                    sampleRate = readLEInt()
                    /* byteRate */ readLEInt()
                    /* blockAlign */ readLEShort()
                    bitsPerSample = readLEShort()
                    val remaining = size - 16
                    if (remaining > 0) raf.seek(raf.filePointer + remaining)
                }
                "data" -> {
                    if (size < 0 || size > (raf.length() - raf.filePointer)) {
                        throw IllegalStateException("Invalid WAV data chunk size: $size")
                    }
                    dataBytes = ByteArray(size)
                    raf.readFully(dataBytes)
                }
                else -> {
                    // Skip unknown chunk (pad to even)
                    val skip = if (size % 2 == 1) size + 1 else size
                    raf.seek(raf.filePointer + skip)
                }
            }
            if (dataBytes != null && sampleRate > 0 && numChannels > 0 && bitsPerSample > 0) break
        }

        require(bitsPerSample == 16) { "WAV must be 16-bit PCM" }
        require(audioFormat == 1) { "Not PCM" }
        require(dataBytes != null) { "No data chunk found" }

        val buf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        val totalSamples = dataBytes!!.size / 2
        val raw = FloatArray(totalSamples)
        var i = 0
        while (buf.hasRemaining()) {
            raw[i++] = (buf.short.toFloat() / 32768.0f)
        }
        val mono = if (numChannels == 1) raw else deinterleaveAndMono(raw, numChannels)
        return if (sampleRate == 22050) mono else resampleLinear(mono, sampleRate, 22050)
    }
}

// Fallback decoder for non-PCM WAV files using Java AudioSystem (desktop equivalent of MediaCodec)
private fun decodeWavWithAudioSystem(path: String): FloatArray {
    val audioInputStream = AudioSystem.getAudioInputStream(File(path))
    val format = audioInputStream.format
    val targetFormat = AudioFormat(
        AudioFormat.Encoding.PCM_SIGNED,
        22050f,
        16,
        1,
        2,
        22050f,
        false
    )
    
    val convertedStream = if (format != targetFormat) {
        AudioSystem.getAudioInputStream(targetFormat, audioInputStream)
    } else {
        audioInputStream
    }
    
    val bytes = convertedStream.readAllBytes()
    val samples = ShortArray(bytes.size / 2)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(samples)
    
    val floats = FloatArray(samples.size)
    for (i in samples.indices) {
        floats[i] = samples[i] / 32768.0f
    }
    
    convertedStream.close()
    audioInputStream.close()
    
    return floats
}

// Viridis-like colormap (matches Flutter app _SpectrogramPainter._colormap)
// Blue -> Cyan -> Yellow gradient
private fun colormapViridis(v: Float): Int {
    val x = v.coerceIn(0f, 1f)
    val (r, g, b) = if (x < 0.5f) {
        // Blue to Cyan
        val t = x / 0.5f
        val r = 0
        val g = (255 * t).toInt().coerceIn(0, 255)
        val b = ((128 * (1 - t) + 255 * t)).toInt().coerceIn(0, 255)
        Triple(r, g, b)
    } else {
        // Cyan to Yellow
        val t = (x - 0.5f) / 0.5f
        val r = (255 * t).toInt().coerceIn(0, 255)
        val g = 255
        val b = ((255 * (1 - t))).toInt().coerceIn(0, 255)
        Triple(r, g, b)
    }
    return (255 shl 24) or (r shl 16) or (g shl 8) or b
}

private fun deinterleaveAndMono(interleaved: FloatArray, channels: Int): FloatArray {
    val totalFrames = interleaved.size / channels
    val out = FloatArray(totalFrames)
    var idx = 0
    for (f in 0 until totalFrames) {
        var sum = 0f
        for (c in 0 until channels) sum += interleaved[idx++]
        out[f] = sum / channels
    }
    return out
}

private fun resampleLinear(x: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
    if (srcRate == dstRate) return x
    val ratio = dstRate.toDouble() / srcRate
    val outLen = kotlin.math.max(1, (x.size * ratio).toInt())
    val out = FloatArray(outLen)
    for (i in 0 until outLen) {
        val pos = i / ratio
        val i0 = kotlin.math.floor(pos).toInt()
        val i1 = kotlin.math.min(i0 + 1, x.size - 1)
        val w = (pos - i0).toFloat()
        out[i] = x[i0] * (1 - w) + x[i1] * w
    }
    return out
}

private fun hannWindow(n: Int): FloatArray {
    val w = FloatArray(n)
    for (i in 0 until n) {
        w[i] = (0.5f - 0.5f * cos(2.0 * PI * i / n).toFloat())
    }
    return w
}

// STFT magnitude squared - matches MainActivity.kt exactly
private fun stftMagnitudeSquared(x: FloatArray, nFft: Int, hop: Int): Array<FloatArray> {
    val win = hannWindow(nFft)
    // center=True equivalent: pad by nFft/2 on both sides
    val pad = nFft / 2
    val padded = FloatArray(x.size + 2 * pad)
    // reflect padding (like librosa default)
    // center segment
    System.arraycopy(x, 0, padded, pad, x.size)
    // left reflect: padded[0..pad-1] = x[pad..1] (no edge repetition)
    var i = 0
    while (i < pad) {
        padded[pad - 1 - i] = x[i + 1]
        i++
    }
    // right reflect: padded[pad + x.size .. end-1] = x[n-2 .. n-pad-1]
    i = 0
    while (i < pad) {
        padded[pad + x.size + i] = x[x.size - 2 - i]
        i++
    }
    val nFrames = 1 + ((padded.size - nFft).coerceAtLeast(0) / hop)
    val fft = FloatFFT_1D(nFft.toLong())
    val spec = Array(nFrames) { FloatArray(nFft / 2 + 1) }
    val frameBuf = FloatArray(nFft * 2)
    var frame = 0
    var start = 0
    while (frame < nFrames) {
        // zero
        java.util.Arrays.fill(frameBuf, 0f)
        var i = 0
        while (i < nFft && start + i < padded.size) {
            frameBuf[2 * i] = padded[start + i] * win[i]
            frameBuf[2 * i + 1] = 0f
            i++
        }
        fft.complexForward(frameBuf)
        for (k in 0..(nFft / 2)) {
            val re = frameBuf[2 * k]
            val im = frameBuf[2 * k + 1]
            spec[frame][k] = re * re + im * im
        }
        frame++
        start += hop
    }
    return spec
}

// Slaney mel scale (matches librosa default when htk=False) - matches MainActivity.kt exactly
private fun hzToMel(hz: Double): Double {
    val fSp = 200.0 / 3.0
    val minLogHz = 1000.0
    val minLogMel = minLogHz / fSp
    val logStep = ln(6.4) / 27.0
    return if (hz >= minLogHz) {
        minLogMel + ln(hz / minLogHz) / logStep
    } else {
        hz / fSp
    }
}

private fun melToHz(mel: Double): Double {
    val fSp = 200.0 / 3.0
    val minLogHz = 1000.0
    val minLogMel = minLogHz / fSp
    val logStep = ln(6.4) / 27.0
    return if (mel >= minLogMel) {
        minLogHz * kotlin.math.exp((mel - minLogMel) * logStep)
    } else {
        mel * fSp
    }
}

private fun log10(x: Double): Double = ln(x) / ln(10.0)

// Mel filter bank with Slaney normalization - matches MainActivity.kt exactly
private fun melFilterBank(sr: Int, nFft: Int, nMels: Int): Array<FloatArray> {
    // Frequency grid for rfft bins [0..nFft/2]
    val numBins = nFft / 2 + 1
    val fftFreqs = DoubleArray(numBins) { k -> (sr.toDouble() / nFft) * k }

    // Slaney mel points
    val fMin = 0.0
    val fMax = sr / 2.0
    val mMin = hzToMel(fMin)
    val mMax = hzToMel(fMax)
    val mPoints = DoubleArray(nMels + 2) { i -> mMin + (mMax - mMin) * i / (nMels + 1) }
    val fPoints = DoubleArray(nMels + 2) { i -> melToHz(mPoints[i]) }

    val fb = Array(nMels) { FloatArray(numBins) }
    for (m in 1..nMels) {
        val fLeft = fPoints[m - 1]
        val fCenter = fPoints[m]
        val fRight = fPoints[m + 1]
        val invLeft = 1.0 / (fCenter - fLeft).coerceAtLeast(1e-12)
        val invRight = 1.0 / (fRight - fCenter).coerceAtLeast(1e-12)
        var k = 0
        while (k < numBins) {
            val fk = fftFreqs[k]
            val w = when {
                fk < fLeft || fk > fRight -> 0.0
                fk < fCenter -> (fk - fLeft) * invLeft
                else -> (fRight - fk) * invRight
            }
            fb[m - 1][k] = w.toFloat()
            k++
        }
    }
    // Slaney normalization: equal area per filter (2 / (f_{m+1}-f_{m-1}))
    for (m in 0 until nMels) {
        val fLeft = fPoints[m]
        val fRight = fPoints[m + 2]
        val enorm = if (fRight > fLeft) (2.0 / (fRight - fLeft)) else 0.0
        var k = 0
        while (k < numBins) {
            fb[m][k] = (fb[m][k] * enorm).toFloat()
            k++
        }
    }
    return fb
}

// Compute mel spectrogram - matches MainActivity.kt exactly
private fun computeMelSpectrogram(
    x: FloatArray,
    sr: Int,
    nMels: Int,
    nFft: Int,
    hop: Int
): Array<FloatArray> {
    val powerSpec = stftMagnitudeSquared(x, nFft, hop)
    val fb = melFilterBank(sr, nFft, nMels)
    val nFrames = powerSpec.size
    val mel = Array(nMels) { FloatArray(nFrames) }
    val eps = 1e-10f
    for (t in 0 until nFrames) {
        for (m in 0 until nMels) {
            var s = 0.0
            val fbRow = fb[m]
            val ps = powerSpec[t]
            val limit = min(ps.size, fbRow.size)
            var k = 0
            while (k < limit) {
                s += ps[k] * fbRow[k]
                k++
            }
            mel[m][t] = s.toFloat()
        }
    }
    // power_to_db(ref=max)
    var maxVal = -Float.MAX_VALUE
    for (m in 0 until nMels) {
        for (t in 0 until nFrames) {
            if (mel[m][t] > maxVal) maxVal = mel[m][t]
        }
    }
    val ref = max(maxVal, eps)
    for (m in 0 until nMels) {
        for (t in 0 until nFrames) {
            val v = mel[m][t]
            val db = 10.0 * log10((v.toDouble() + eps) / ref.toDouble())
            mel[m][t] = db.toFloat()
        }
    }
    return mel
}

// Bilinear resize from source [Hsrc=nMels][Wsrc=time] to [Htgt][Wtgt] - matches MainActivity.kt exactly
private fun resizeBilinear(src: Array<FloatArray>, hTgt: Int, wTgt: Int): Array<FloatArray> {
    val hSrc = src.size
    val wSrc = if (hSrc > 0) src[0].size else 0
    val out = Array(hTgt) { FloatArray(wTgt) }
    if (hSrc == 0 || wSrc == 0) return out

    val yScale = (hSrc - 1).toFloat() / (hTgt - 1).coerceAtLeast(1)
    val xScale = (wSrc - 1).toFloat() / (wTgt - 1).coerceAtLeast(1)
    for (y in 0 until hTgt) {
        val srcY = y * yScale
        val y0 = kotlin.math.floor(srcY).toInt()
        val y1 = kotlin.math.min(y0 + 1, hSrc - 1)
        val wy = srcY - y0
        for (x in 0 until wTgt) {
            val srcX = x * xScale
            val x0 = kotlin.math.floor(srcX).toInt()
            val x1 = kotlin.math.min(x0 + 1, wSrc - 1)
            val wx = srcX - x0

            val v00 = src[y0][x0]
            val v01 = src[y0][x1]
            val v10 = src[y1][x0]
            val v11 = src[y1][x1]
            val top = v00 * (1 - wx) + v01 * wx
            val bot = v10 * (1 - wx) + v11 * wx
            out[y][x] = (top * (1 - wy) + bot * wy).toFloat()
        }
    }
    return out
}