package com.example.congressionalapp

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.transform.*
import java.io.*
import java.nio.*
import kotlin.math.*
import android.media.*
import java.nio.*

class MainActivity : FlutterActivity() {

    private val CHANNEL = "parkinsons/audio"
    private var module: Module? = null

    // ---- Librosa-compatible params ----
    private val SAMPLE_RATE = 22050
    private val N_FFT = 2048
    private val HOP_LENGTH = 512
    private val WIN_LENGTH = 2048
    private val N_MELS = 128
    private val FMIN = 0.0
    private val FMAX = SAMPLE_RATE / 2.0
    private val AMIN = 1e-10
    private val TOP_DB = 80.0

    // ImageNet normalization constants
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val INPUT_SIZE = 299

    // Hann window
    private val HANN: DoubleArray by lazy {
        DoubleArray(WIN_LENGTH) { i ->
            0.5 * (1.0 - cos(2.0 * Math.PI * i / (WIN_LENGTH - 1)))
        }
    }
    

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                if (call.method == "analyzeFile") {
                    val path = call.argument<String>("path")
                    if (path == null) {
                        result.error("ARG", "Missing file path", null)
                        return@setMethodCallHandler
                    }

                    try {
                        if (module == null)
                            module = Module.load(assetFilePath("best_model.pt"))

                        val audio = readAudioMono(path)
                        val melDb = computeLogMelLibrosaExact(audio)
                        val input = makeInputTensorFromMel(melDb)

                        val out = module!!.forward(IValue.from(input)).toTensor()
                        val scores = out.dataAsFloatArray
                        val pdProb =
                            if (scores.size == 1) sigmoid(scores[0]) else softmax(scores)[1]

                        val cacheFile = File(cacheDir, "mel_spectrogram.png")
                        melToBitmap(melDb).compress(Bitmap.CompressFormat.PNG, 100, FileOutputStream(cacheFile))
                        result.success(mapOf("prob_pd" to pdProb, "image_path" to cacheFile.absolutePath))

                    } catch (e: Exception) {
                        Log.e("MainActivity", "Model error: ${e.message}", e)
                        result.error("ERR", e.message, null)
                    }
                } else {
                    result.notImplemented()
                }
            }
    }

    // === WAV reader (16-bit mono PCM) ===
    private fun readAudioMono(path: String): FloatArray {
        return when {
            path.lowercase().endsWith(".wav") -> readWavMonoRobust(path)
            else -> decodeWithMediaCodec(path)
        }
    }


    // === Librosa-equivalent mel-spectrogram ===
    private fun computeLogMelLibrosaExact(wav: FloatArray): Array<FloatArray> {
        val pad = WIN_LENGTH / 2
        val centered = reflectPad(wav, pad)
        val stftPow = stftPower(centered, N_FFT, HOP_LENGTH, WIN_LENGTH, HANN)
        val melFb = melFilterBankSlaney(N_MELS, N_FFT, SAMPLE_RATE.toDouble(), FMIN, FMAX)
        val mel = dot(melFb, stftPow)
        val melDb = powerToDb(mel, AMIN, TOP_DB)
        return Array(melDb.size) { m ->
            FloatArray(melDb[0].size) { t -> melDb[m][t].toFloat() }
        }
    }

    // === Convert mel spectrogram → normalized RGB tensor ===
    private fun makeInputTensorFromMel(melDb: Array<FloatArray>): Tensor {
        val bmp = melToBitmap(melDb)
        val chw = TensorImageUtils.bitmapToFloat32Tensor(bmp, IMAGENET_MEAN, IMAGENET_STD)
        val data = chw.dataAsFloatArray
        return Tensor.fromBlob(data, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    private fun melToBitmap(melDb: Array<FloatArray>): Bitmap {
        val h = melDb.size
        val w = melDb[0].size
        var minV = Float.POSITIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        for (y in 0 until h) for (x in 0 until w) {
            val v = melDb[y][x]
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }
        val range = (maxV - minV).coerceAtLeast(1e-6f)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val norm = ((melDb[y][x] - minV) / range).coerceIn(0f, 1f)
                val g = (norm * 255f + 0.5f).toInt().coerceIn(0, 255)
                val color = Color.rgb(g, g, g)
                bmp.setPixel(x, y, color)
            }
        }
        return Bitmap.createScaledBitmap(bmp, INPUT_SIZE, INPUT_SIZE, true)
    }

    // === DSP Utility functions ===
    private fun reflectPad(x: FloatArray, pad: Int): DoubleArray {
        val n = x.size
        val out = DoubleArray(n + 2 * pad)
        for (i in 0 until pad) out[pad - 1 - i] = x[min(i, n - 1)].toDouble()
        for (i in 0 until n) out[pad + i] = x[i].toDouble()
        for (i in 0 until pad) out[pad + n + i] = x[max(n - 2 - i, 0)].toDouble()
        return out
    }

    private fun stftPower(
        x: DoubleArray,
        nFft: Int,
        hop: Int,
        winLen: Int,
        window: DoubleArray
    ): Array<DoubleArray> {
        val fft = FastFourierTransformer(DftNormalization.UNITARY)
        val nFrames = 1 + (x.size - winLen).coerceAtLeast(0) / hop
        val pow = Array(nFft / 2 + 1) { DoubleArray(nFrames) }
        val frame = DoubleArray(nFft)
        for (t in 0 until nFrames) {
            val start = t * hop
            java.util.Arrays.fill(frame, 0.0)
            for (i in 0 until winLen) frame[i] = x[start + i] * window[i]
            val complex: Array<Complex> = fft.transform(frame, TransformType.FORWARD)
            for (k in 0..nFft / 2) {
                val c = complex[k]
                pow[k][t] = c.real * c.real + c.imaginary * c.imaginary
            }
        }
        return pow
    }

    private fun dot(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val m = A.size
        val k = A[0].size
        val n = B[0].size
        val C = Array(m) { DoubleArray(n) }
        for (i in 0 until m) {
            val Ai = A[i]
            for (j in 0 until n) {
                var s = 0.0
                for (p in 0 until k) s += Ai[p] * B[p][j]
                C[i][j] = s
            }
        }
        return C
    }

    private fun melFilterBankSlaney(
        nMels: Int,
        nFft: Int,
        sr: Double,
        fmin: Double,
        fmax: Double
    ): Array<DoubleArray> {
        fun hzToMel(f: Double): Double = 2595.0 * ln(1.0 + f / 700.0)
        fun melToHz(m: Double): Double = 700.0 * (exp(m / 2595.0) - 1.0)

        val mMin = hzToMel(fmin)
        val mMax = hzToMel(fmax)
        val mPts = DoubleArray(nMels + 2) { i -> mMin + (mMax - mMin) * i / (nMels + 1) }
        val fPts = DoubleArray(nMels + 2) { i -> melToHz(mPts[i]) }
        val fftFreqs = DoubleArray(nFft / 2 + 1) { i -> (sr / nFft) * i }
        val fb = Array(nMels) { DoubleArray(nFft / 2 + 1) }

        for (m in 1..nMels) {
            val f0 = fPts[m - 1]; val f1 = fPts[m]; val f2 = fPts[m + 1]
            for (i in fftFreqs.indices) {
                val f = fftFreqs[i]
                val w = when {
                    f < f0 || f > f2 -> 0.0
                    f < f1 -> (f - f0) / (f1 - f0).coerceAtLeast(1e-12)
                    else -> (f2 - f) / (f2 - f1).coerceAtLeast(1e-12)
                }
                fb[m - 1][i] = w
            }
        }

        // Slaney normalization
        for (m in 0 until nMels) {
            val sum = fb[m].sum().coerceAtLeast(1e-12)
            for (i in fb[m].indices) fb[m][i] /= sum
        }
        return fb
    }

    private fun powerToDb(S: Array<DoubleArray>, amin: Double, topDb: Double): Array<DoubleArray> {
        val m = S.size
        val n = S[0].size
        var maxVal = amin
        for (i in 0 until m) for (j in 0 until n) maxVal = max(maxVal, S[i][j])
        val ref = maxVal.coerceAtLeast(amin)

        val out = Array(m) { DoubleArray(n) }
        var maxDb = -Double.MAX_VALUE
        for (i in 0 until m) {
            for (j in 0 until n) {
                val v = max(S[i][j], amin)
                val db = 10.0 * log10(v / ref)
                out[i][j] = db
                if (db > maxDb) maxDb = db
            }
        }
        val lower = maxDb - topDb
        for (i in 0 until m) for (j in 0 until n)
            if (out[i][j] < lower) out[i][j] = lower
        return out
    }

    // === Math helpers ===
    private fun softmax(x: FloatArray): DoubleArray {
        val mx = x.maxOrNull() ?: 0f
        val exps = x.map { exp((it - mx).toDouble()) }
        val sum = exps.sum()
        return exps.map { it / sum }.toDoubleArray()
    }

    private fun sigmoid(v: Float): Double = 1.0 / (1.0 + exp(-v.toDouble()))

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }

    private fun decodeWithMediaCodec(path: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)

        var trackIndex = -1
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME)
            if (mime != null && mime.startsWith("audio/")) {
                trackIndex = i
                extractor.selectTrack(i)
                break
            }
        }
        if (trackIndex == -1) throw IllegalArgumentException("No audio track found in $path")

        val inputFormat = extractor.getTrackFormat(trackIndex)
        val sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(inputFormat.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(inputFormat, null, null, 0)
        codec.start()

        val bufSize = 262144 // 256 KB internal buffer
        val outputBuffer = ArrayList<Float>()
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inIndex = codec.dequeueInputBuffer(10000)
                if (inIndex >= 0) {
                    val inputBuf = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inputBuf, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        val pts = extractor.sampleTime
                        codec.queueInputBuffer(inIndex, 0, sampleSize, pts, 0)
                        extractor.advance()
                    }
                }
            }

            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10000)
            when {
                outIndex >= 0 -> {
                    val outBuf = codec.getOutputBuffer(outIndex)!!
                    if (bufferInfo.size > 0) {
                        val shortBuf = outBuf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        val frame = ShortArray(shortBuf.remaining())
                        shortBuf.get(frame)
                        for (i in frame.indices step channels) {
                            var sum = 0f
                            for (ch in 0 until channels) {
                                if (i + ch < frame.size)
                                    sum += frame[i + ch] / 32768f
                            }
                            outputBuffer.add(sum / channels)
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0)
                        sawOutputEOS = true
                }
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    // Format might change midstream — can be ignored safely
                }
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        val pcm = outputBuffer.toFloatArray()
        return if (sampleRate != SAMPLE_RATE) resampleLinear(pcm, sampleRate, SAMPLE_RATE) else pcm
    }
// Usage: val pcm = readWavMonoRobust(path)
private fun readWavMonoRobust(path: String): FloatArray {
    RandomAccessFile(path, "r").use { raf ->
        // --- Basic RIFF/WAVE validation ---
        val header = ByteArray(12)
        raf.readFully(header)
        val riff = String(header, 0, 4)
        val wave = String(header, 8, 4)
        require(riff == "RIFF" && wave == "WAVE") { "Invalid WAV" }

        var numChannels = 1
        var sampleRate = SAMPLE_RATE
        var bitsPerSample = 16
        var dataPos = -1L
        var dataSize = -1
        var audioFormat = 1 // PCM default

        // --- Parse chunks ---
        while (raf.filePointer + 8 < raf.length()) {
            val idBytes = ByteArray(4)
            val sizeBytes = ByteArray(4)
            raf.readFully(idBytes)
            raf.readFully(sizeBytes)
            val id = String(idBytes)
            val size = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).int

            when (id) {
                "fmt " -> {
                    val fmt = ByteArray(size)
                    raf.readFully(fmt)
                    val bb = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
                    audioFormat = bb.short.toInt()
                    numChannels = bb.short.toInt()
                    sampleRate = bb.int
                    bb.int // byteRate
                    bb.short // blockAlign
                    bitsPerSample = bb.short.toInt()
                }
                "data" -> {
                    dataPos = raf.filePointer
                    dataSize = size
                    break
                }
                else -> raf.seek(raf.filePointer + size)
            }
            if (size % 2 == 1) raf.seek(raf.filePointer + 1)
        }
        require(dataPos >= 0 && dataSize > 0) { "No data chunk" }

        // --- Stream decode directly to FloatArray ---
        raf.seek(dataPos)
        val bytesPerSample = (bitsPerSample + 7) / 8
        val frameSize = bytesPerSample * numChannels
        val totalFrames = dataSize / frameSize
        val maxFrames = minOf(totalFrames, 30 * sampleRate) // limit ~30 s audio
        val pcm = FloatArray(maxFrames)

        val tmp = ByteArray(frameSize)
        val bb = ByteBuffer.wrap(tmp).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until maxFrames) {
            raf.readFully(tmp)
            bb.clear()
            var acc = 0f
            repeat(numChannels) {
                val sample = when (bitsPerSample) {
                    8 -> ((bb.get().toInt() and 0xff) - 128) / 128f
                    16 -> bb.short / 32768f
                    24 -> {
                        val b0 = bb.get().toInt() and 0xff
                        val b1 = bb.get().toInt() and 0xff
                        val b2 = bb.get().toInt() and 0xff
                        var v = (b0 or (b1 shl 8) or (b2 shl 16))
                        if (v and 0x800000 != 0) v = v or -0x1000000
                        v / 8388608f
                    }
                    32 -> bb.int / 2147483648f
                    else -> 0f
                }
                acc += sample
            }
            pcm[i] = (acc / numChannels).coerceIn(-1f, 1f)
        }

        return if (sampleRate != SAMPLE_RATE)
            resampleLinear(pcm, sampleRate, SAMPLE_RATE)
        else pcm
    }
}

// --- Simple linear resampler (mono float) ---
    private fun resampleLinear(input: FloatArray, oldRate: Int, newRate: Int): FloatArray {
        if (oldRate == newRate) return input
        if (input.isEmpty()) return input

        val ratio = newRate.toDouble() / oldRate
        val newLength = max(1, (input.size * ratio).toInt())
        val output = FloatArray(newLength)

        for (i in 0 until newLength) {
            val srcPos = i / ratio
            val idx0 = floor(srcPos).toInt().coerceIn(0, input.size - 1)
            val idx1 = (idx0 + 1).coerceAtMost(input.size - 1)
            val frac = (srcPos - idx0).toFloat()
            output[i] = (1 - frac) * input[idx0] + frac * input[idx1]
        }
        return output
    }



}
