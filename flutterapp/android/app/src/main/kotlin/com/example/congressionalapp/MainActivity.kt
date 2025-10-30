package com.example.congressionalapp

import io.flutter.embedding.android.FlutterActivity
import io.flutter.plugin.common.MethodChannel
import org.pytorch.Module
import org.pytorch.IValue
import org.pytorch.Tensor
import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.IOException
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.pow
import org.jtransforms.fft.FloatFFT_1D
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.RandomAccessFile
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaCodec

class MainActivity : FlutterActivity() {
    private val CHANNEL = "pd_infer"
    private var torchModule: Module? = null

    override fun configureFlutterEngine(flutterEngine: io.flutter.embedding.engine.FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "analyzeFile" -> {
                        val path = call.argument<String>("path")
                        if (path == null) {
                            result.error("ARG", "Missing path", null)
                            return@setMethodCallHandler
                        }
                        try {
                            val resultData = analyzeWithKotlinAndTorch(path)
                            val label = if (resultData.probability >= 0.5f) "PD" else "HC"
                            val melList = toNestedList(resultData.mel)
                            val map = hashMapOf(
                                "label" to label,
                                "probability" to resultData.probability.toDouble(),
                                "mel" to melList
                            )
                            result.success(map)
                        } catch (e: Exception) {
                            result.error("ERR", e.toString(), null)
                        }
                    }
                    "exportMelCsv" -> {
                        val path = call.argument<String>("path")
                        val fileName = call.argument<String>("fileName") ?: "mel_dump.csv"
                        if (path == null) {
                            result.error("ARG", "Missing path", null)
                            return@setMethodCallHandler
                        }
                        try {
                            val sr = 22050
                            val pcm = try { readWavPcmAsMono16k(path) } catch (_: Exception) { decodeToMono16k(path) }
                            val mel = computeMelSpectrogram(pcm, sr, nMels = 128, nFft = 2048, hop = 512)
                            val outFile = File(applicationContext.filesDir, fileName)
                            writeMelCsv(mel, outFile)
                            result.success(outFile.absolutePath)
                        } catch (e: Exception) {
                            result.error("ERR", e.toString(), null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    data class AnalysisResult(val probability: Float, val mel: Array<FloatArray>)

    private fun analyzeWithKotlinAndTorch(audioPath: String): AnalysisResult {
        // Try fast-path WAV PCM; if not PCM or fails, fall back to decoder.
        val pcm = try {
            readWavPcmAsMono16k(audioPath)
        } catch (_: Exception) {
            decodeToMono16k(audioPath)
        }
        val sr = 22050
        val mel = computeMelSpectrogram(pcm, sr, nMels = 128, nFft = 2048, hop = 512)
        val nMels = mel.size
        val timeSteps = if (nMels > 0) mel[0].size else 0

        // Inception expects 3 channels and 299x299.
        val targetH = 299
        val targetW = 299
        val resized = resizeBilinear(mel, targetH, targetW) // [H][W]
        // Min-max scale to [0,1] to reduce distribution shift versus training images
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
        // Make 3 channels by replication
        val data = FloatArray(3 * targetH * targetW)
        var idx = 0
        for (c in 0 until 3) {
            for (i in 0 until targetH) {
                for (j in 0 until targetW) {
                    val norm = (resized[i][j] - mn) / range
                    data[idx++] = norm
                }
            }
        }

        if (torchModule == null) {
            torchModule = loadTorchModuleFromFlutterAssets(this, "assets/models/best_model.pt")
        }
        val moduleTorch = requireNotNull(torchModule)

        val shape = longArrayOf(1, 3, targetH.toLong(), targetW.toLong())
        val inputTensor = Tensor.fromBlob(data, shape)
        val out = moduleTorch.forward(IValue.from(inputTensor)).toTensor()
        val outData = out.dataAsFloatArray
        val prob: Float = if (outData.size <= 1) {
            val logit = if (outData.isNotEmpty()) outData[0] else 0f
            (1f / (1f + (-logit).let { exp(it.toDouble()).toFloat() }))
        } else {
            // 2-class (or multi-class) softmax; assume index 1 corresponds to PD
            val maxLogit = outData.maxOrNull() ?: 0f
            var sum = 0.0
            val exps = DoubleArray(outData.size) { k -> kotlin.math.exp((outData[k] - maxLogit).toDouble()) }
            for (v in exps) sum += v
            val pdIdx = if (outData.size > 1) 1 else 0
            (exps[pdIdx] / sum).toFloat()
        }
        return AnalysisResult(prob, mel)
    }

    private fun loadTorchModuleFromFlutterAssets(context: Context, assetKey: String): Module {
        val loader = io.flutter.FlutterInjector.instance().flutterLoader()
        val lookup = loader.getLookupKeyForAsset(assetKey)
        val input: InputStream = try {
            context.assets.open(lookup)
        } catch (ex: Exception) {
            throw IllegalStateException("Asset not found: $assetKey (lookup=$lookup). Ensure it is listed under flutter/assets in pubspec.yaml.")
        }
        val file = File(context.filesDir, "best_model.pt")
        FileOutputStream(file).use { out ->
            input.copyTo(out)
        }
        return Module.load(file.absolutePath)
    }

    // Reads WAV (PCM 16-bit) and converts to mono 16 kHz. If not PCM, throws.
    private fun readWavPcmAsMono16k(path: String): FloatArray {
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
                        // Don't require here; if not PCM we'll fallback to decoder at caller
                    }
                    "data" -> {
                        if (size < 0 || size > (raf.length() - raf.filePointer)) {
                            throw IOException("Invalid WAV data chunk size: $size")
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

            require(bitsPerSample == 16) { "WAV must be 16-bit PCM or use decoder" }
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

    // Decode arbitrary audio (e.g., compressed WAV, MP3, M4A) to mono 16 kHz via MediaCodec
    private fun decodeToMono16k(path: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            val mime = fmt.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) { trackIndex = i; break }
        }
        require(trackIndex >= 0) { "No audio track found" }
        extractor.selectTrack(trackIndex)
        val format = extractor.getTrackFormat(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val outSamples = ArrayList<Short>()
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        while (!outputDone) {
            if (!inputDone) {
                val inIndex = codec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val inBuf = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inBuf, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        val pts = extractor.sampleTime
                        codec.queueInputBuffer(inIndex, 0, sampleSize, pts, 0)
                        extractor.advance()
                    }
                }
            }
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {}
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {}
                outIndex >= 0 -> {
                    val outBuf = codec.getOutputBuffer(outIndex)!!
                    val bytes = ByteArray(bufferInfo.size)
                    outBuf.get(bytes)
                    outBuf.clear()
                    codec.releaseOutputBuffer(outIndex, false)

                    // Assume PCM 16-bit
                    val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    while (bb.remaining() >= 2) {
                        outSamples.add(bb.short)
                    }
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        outputDone = true
                    }
                }
            }
        }
        codec.stop(); codec.release(); extractor.release()
        // Convert to float, mix to mono if needed
        val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val floats: FloatArray = if (channelCount <= 1) {
            FloatArray(outSamples.size) { i -> outSamples[i] / 32768.0f }
        } else {
            val totalFrames = outSamples.size / channelCount
            val mono = FloatArray(totalFrames)
            var idx = 0
            for (f in 0 until totalFrames) {
                var sum = 0f
                for (c in 0 until channelCount) {
                    sum += outSamples[idx++]/32768.0f
                }
                mono[f] = sum / channelCount
            }
            mono
        }
        return if (srcRate == 22050) floats else resampleLinear(floats, srcRate, 22050)
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

    // Slaney mel scale (matches librosa default when htk=False)
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

    // Bilinear resize from source [Hsrc=nMels][Wsrc=time] to [Htgt][Wtgt]
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

    private fun toNestedList(mel: Array<FloatArray>): ArrayList<ArrayList<Double>> {
        val nMels = mel.size
        val timeSteps = if (nMels > 0) mel[0].size else 0
        val outer = ArrayList<ArrayList<Double>>(nMels)
        for (m in 0 until nMels) {
            val row = ArrayList<Double>(timeSteps)
            for (t in 0 until timeSteps) {
                row.add(mel[m][t].toDouble())
            }
            outer.add(row)
        }
        return outer
    }

    private fun writeMelCsv(mel: Array<FloatArray>, file: File) {
        val sb = StringBuilder()
        val nMels = mel.size
        val nFrames = if (nMels > 0) mel[0].size else 0
        // Header: parameters used for reproducibility
        sb.append("sr,22050\n")
        sb.append("n_fft,2048\n")
        sb.append("hop_length,512\n")
        sb.append("n_mels,128\n")
        sb.append("center,True\n")
        sb.append("mel_scale,slaney\n")
        sb.append("norm,slaney\n")
        sb.append("rows=mel_bands,cols=time_frames\n")
        // Data: each line is one mel band across time
        for (m in 0 until nMels) {
            var t = 0
            while (t < nFrames) {
                sb.append(mel[m][t])
                if (t < nFrames - 1) sb.append(',')
                t++
            }
            sb.append('\n')
        }
        FileOutputStream(file).use { fos ->
            fos.write(sb.toString().toByteArray(Charsets.UTF_8))
        }
    }
}
