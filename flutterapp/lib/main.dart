// DISCLAIMER: Research and educational use only — not a medical tool.

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/services.dart';

void main() => runApp(const ParkinsonsApp());

class ParkinsonsApp extends StatelessWidget {
  const ParkinsonsApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Parkinson’s Voice Analyzer',
        theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo)),
        home: const HomeScreen(),
      );
}


class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/logo.png',
                height: 32,
                width: 32,
                errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white),
              ),
              const SizedBox(width: 12),
              const Text("Parkinson's Voice Analyzer"),
            ],
          ),
          centerTitle: true,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Column(
                  children: [
                    Image.asset(
                      'assets/logo.png',
                      height: 120,
                      width: 120,
                      errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, size: 100, color: Colors.indigo),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Neuro Sense',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.indigo,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      "Parkinson's Voice Analyzer",
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.black54,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 40),

                
                ElevatedButton.icon(
                  icon: const Icon(Icons.mic, color: Colors.white),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const RecordScreen()),
                  ),
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(250, 60),
                    backgroundColor: Colors.indigo,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  ),
                  label: const Text(
                    'Record Audio',
                    style: TextStyle(fontSize: 18, color: Colors.white),
                  ),
                ),
                const SizedBox(height: 20),

                
                ElevatedButton.icon(
                  icon: const Icon(Icons.upload_file, color: Colors.white),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const UploadScreen()),
                  ),
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(250, 60),
                    backgroundColor: Colors.indigo,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  ),
                  label: const Text(
                    'Upload Audio File',
                    style: TextStyle(fontSize: 18, color: Colors.white),
                  ),
                ),
                const SizedBox(height: 20),

                
                TextButton(
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const InfoScreen()),
                  ),
                  child: const Text(
                    'Learn About Parkinson’s',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.indigo,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),

                
                const SizedBox(height: 40),
                const Divider(thickness: 1),
                const SizedBox(height: 10),
                const Text(
                  'This application is for research and educational use only.\n'
                  'It is not intended for medical diagnosis or treatment.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13,
                    height: 1.3,
                    color: Colors.black54,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
}


class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});
  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final _recorder = AudioRecorder();
  bool _isRecording = false;
  String _status = 'Tap the mic to start recording';
  String? _savedFilePath;
  Timer? _timer;
  int _seconds = 0;

  static const _channel = MethodChannel('pd_infer');

  Future<void> _analyzeAndShow(File file) async {
    setState(() => _status = 'Analyzing on-device...');
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('analyzeFile', {
        'path': file.path,
      });
      if (!mounted) return;
      final label = res?['label'];
      final prob = (res?['probability'] as num?)?.toDouble();
      final mel = (res?['mel'] as List?)?.cast<List>().map((r) => r.cast<num>().map((e) => e.toDouble()).toList()).toList();
      
      if (mel != null && label != null && prob != null) {
        if (!mounted) return;
        Navigator.of(context).push(MaterialPageRoute(
          builder: (_) => ResultsScreen(
            diagnosis: label,
            confidence: prob,
            mel: mel,
          ),
        ));
      }
      setState(() => _status = 'Analysis complete.');
    } on PlatformException catch (e) {
      setState(() => _status = 'Error: ${e.message}');
    } catch (e) {
      setState(() => _status = 'Failed: $e');
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      _timer?.cancel();
      setState(() {
        _isRecording = false;
        _savedFilePath = path;
        _status = 'Recording saved to:\n$path';
      });
    } else {
      if (!await Permission.microphone.request().isGranted) {
        setState(() => _status = 'Microphone permission denied');
        return;
      }

      final dir = await getApplicationDocumentsDirectory();
      final filePath =
          '${dir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';

      const cfg = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      );

      await _recorder.start(cfg, path: filePath);
      _startTimer();
      setState(() {
        _isRecording = true;
        _status = 'Recording...';
      });
    }
  }

  void _startTimer() {
    _seconds = 0;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      setState(() => _seconds++);
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/logo.png',
                height: 28,
                width: 28,
                errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 8),
              const Text('Record Audio'),
            ],
          ),
          centerTitle: true,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: SingleChildScrollView(
              child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              const Text(
                'Please say “ahhhh” in a steady voice\nfor 5–10 seconds.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              ),
                  const SizedBox(height: 8),
              const Text(
                'Try to minimize background noise during recording.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.black54,
                  fontStyle: FontStyle.italic,
                ),
              ),

              const SizedBox(height: 20),
              GestureDetector(
                onTap: _toggleRecording,
                child: CircleAvatar(
                  radius: 110,
                  backgroundColor: _isRecording ? Colors.redAccent : Colors.indigo,
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.mic,
                    color: Colors.white,
                    size: 110,
                  ),
                ),
              ),
              const SizedBox(height: 30),
              if (_isRecording)
                Text(
                  'Recording: ${_seconds}s',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                ),
              const SizedBox(height: 20),
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16),
              ),
              if (_savedFilePath != null) ...[
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  icon: const Icon(Icons.analytics),
                  label: const Text('Analyze Recording'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.indigo,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: () {
                    final path = _savedFilePath;
                    if (path != null) _analyzeAndShow(File(path));
                  },
                ),
              ],
              ]),
            ),
          ),
        ),
      );
}

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});
  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  File? _selectedFile;
  static const _channel = MethodChannel('pd_infer');

  Future<void> _analyzeAndShow(File file) async {
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('analyzeFile', {
        'path': file.path,
      });
      if (!mounted) return;
      final label = res?['label'];
      final prob = (res?['probability'] as num?)?.toDouble();
      final mel = (res?['mel'] as List?)?.cast<List>().map((r) => r.cast<num>().map((e) => e.toDouble()).toList()).toList();
      
      if (mel != null && label != null && prob != null) {
        Navigator.of(context).push(MaterialPageRoute(
          builder: (_) => ResultsScreen(
            diagnosis: label,
            confidence: prob,
            mel: mel,
          ),
        ));
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Analysis failed: Missing data')),
        );
      }
    } on PlatformException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    }
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );
    if (result != null && result.files.single.path != null) {
      setState(() => _selectedFile = File(result.files.single.path!));
    }
  }

  Future<void> _exportMelCsv() async {
    try {
      final pathArg = _selectedFile?.path ?? '/sdcard/Download/test.wav';
      final out = await _channel.invokeMethod<String>('exportMelCsv', {
        'path': pathArg,
        'fileName': 'mel_dump.csv',
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Exported mel CSV to:\n$out')),
      );

      print('MEL_CSV_PATH=$out');
    } on PlatformException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export error: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/logo.png',
                height: 28,
                width: 28,
                errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 8),
              const Text('Upload Audio File'),
            ],
          ),
          centerTitle: true,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: SingleChildScrollView(
              child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(Icons.upload_file, size: 100, color: Colors.indigo.shade400),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: _pickFile,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(200, 50),
                  backgroundColor: Colors.indigo,
                ),
                child: const Text('Choose File',
                    style: TextStyle(fontSize: 18, color: Colors.white)),
              ),
              const SizedBox(height: 20),
              if (_selectedFile != null)
                Text(
                  'Selected: ${_selectedFile!.path.split('/').last}',
                  textAlign: TextAlign.center,
                ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _selectedFile != null
                    ? () {
                        final f = _selectedFile;
                        if (f != null) _analyzeAndShow(f);
                      }
                    : null,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(200, 50),
                  backgroundColor: Colors.indigo,
                ),
                child: const Text('Analyze File',
                    style: TextStyle(fontSize: 18, color: Colors.white)),
              ),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _exportMelCsv,
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(200, 50),
                  backgroundColor: Colors.deepPurple,
                ),
                child: const Text('Export Mel CSV',
                    style: TextStyle(fontSize: 18, color: Colors.white)),
              ),
              ]),
            ),
          ),
        ),
      );
}

class ResultsScreen extends StatelessWidget {
  final String diagnosis; 
  final double confidence;
  final List<List<double>> mel; 

  const ResultsScreen({
    super.key,
    required this.diagnosis,
    required this.confidence,
    required this.mel,
  });

  String _getDiagnosisText() {
    if (diagnosis.toUpperCase() == 'PD' || diagnosis.toUpperCase() == 'PARKINSON') {
      return 'Parkinson\'s Detected';
    } else if (diagnosis.toUpperCase() == 'HC' || diagnosis.toUpperCase() == 'HEALTHY') {
      return 'No Parkinson\'s Detected';
    } else {
      return 'Unknown Result';
    }
  }

  Color _getDiagnosisColor() {
    if (diagnosis.toUpperCase() == 'PD' || diagnosis.toUpperCase() == 'PARKINSON') {
      return Colors.orange.shade700;
    } else {
      return Colors.green.shade700;
    }
  }

  IconData _getDiagnosisIcon() {
    if (diagnosis.toUpperCase() == 'PD' || diagnosis.toUpperCase() == 'PARKINSON') {
      return Icons.warning;
    } else {
      return Icons.check_circle;
    }
  }

  @override
  Widget build(BuildContext context) {
    final diagnosisText = _getDiagnosisText();
    final diagnosisColor = _getDiagnosisColor();
    final diagnosisIcon = _getDiagnosisIcon();
    final confidencePercent = (confidence * 100).toStringAsFixed(1);

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'assets/logo.png',
              height: 28,
              width: 28,
              errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 8),
            const Text('Analysis Results'),
          ],
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
                      Icon(diagnosisIcon, size: 64, color: diagnosisColor),
                      const SizedBox(height: 16),
                      Text(
                        diagnosisText,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: diagnosisColor,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Confidence Level',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: LinearProgressIndicator(
                                value: confidence,
                                backgroundColor: Colors.grey.shade300,
                                valueColor: AlwaysStoppedAnimation<Color>(diagnosisColor),
                                minHeight: 24,
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Text(
                            '$confidencePercent%',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: diagnosisColor,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'The model is ${confidencePercent}% confident in this result.',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.black54,
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

            
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Voice Spectrogram',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'This visualization shows the frequency content of your voice over time.',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.black54,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.grey.shade300),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: AspectRatio(
                            aspectRatio: (mel.isNotEmpty ? mel[0].length : 1) / (mel.isEmpty ? 1 : mel.length),
                            child: CustomPaint(
                              painter: _SpectrogramPainter(mel),
                              isComplex: true,
                              willChange: false,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

             
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, color: Colors.red.shade700, size: 24),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'This result is for research and educational purposes only. It is not a medical diagnosis. Please consult a healthcare professional for medical advice.',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.red.shade900,
                          fontStyle: FontStyle.italic,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class SpectrogramScreen extends StatelessWidget {
  final List<List<double>> mel; 
  const SpectrogramScreen({super.key, required this.mel});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'assets/logo.png',
              height: 28,
              width: 28,
              errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 8),
            const Text('Spectrogram'),
          ],
        ),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: AspectRatio(
            aspectRatio: (mel.isNotEmpty ? mel[0].length : 1) / (mel.isEmpty ? 1 : mel.length),
            child: CustomPaint(
              painter: _SpectrogramPainter(mel),
              isComplex: true,
              willChange: false,
            ),
          ),
        ),
      ),
    );
  }
}

class _SpectrogramPainter extends CustomPainter {
  final List<List<double>> mel;
  late final int nMels;
  late final int timeSteps;
  late final double vMin;
  late final double vMax;

  _SpectrogramPainter(this.mel) {
    nMels = mel.length;
    timeSteps = nMels > 0 ? mel[0].length : 0;
    double mn = double.infinity, mx = -double.infinity;
    for (final row in mel) {
      for (final v in row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    vMin = mn.isFinite ? mn : -1;
    vMax = mx.isFinite ? mx : 1;
    if (vMax - vMin < 1e-6) {
      vMin -= 1;
      vMax += 1;
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (nMels == 0 || timeSteps == 0) return;
    final cellW = size.width / timeSteps;
    final cellH = size.height / nMels;
    final paint = Paint()..style = PaintingStyle.fill;

    for (int m = 0; m < nMels; m++) {
      for (int t = 0; t < timeSteps; t++) {
        final v = mel[m][t];
        final x = t * cellW;
        final y = (nMels - 1 - m) * cellH;
        paint.color = _colormap((v - vMin) / (vMax - vMin));
        canvas.drawRect(Rect.fromLTWH(x, y, cellW, cellH), paint);
      }
    }
  }

  
  Color _colormap(double x) {
    final v = x.clamp(0.0, 1.0);
   
    if (v < 0.5) {
      final t = v / 0.5;
      final r = (0 * (1 - t) + 0 * t).toInt();
      final g = (0 * (1 - t) + 255 * t).toInt();
      final b = (128 * (1 - t) + 255 * t).toInt();
      return Color.fromARGB(255, r, g, b);
    } else {
      final t = (v - 0.5) / 0.5;
      final r = (0 * (1 - t) + 255 * t).toInt();
      final g = (255 * (1 - t) + 255 * t).toInt();
      final b = (255 * (1 - t) + 0 * t).toInt();
      return Color.fromARGB(255, r, g, b);
    }
  }

  @override
  bool shouldRepaint(covariant _SpectrogramPainter oldDelegate) => false;
}

class InfoScreen extends StatelessWidget {
  const InfoScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/logo.png',
                height: 28,
                width: 28,
                errorBuilder: (context, error, stackTrace) => const Icon(Icons.biotech, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 8),
              const Text("About Parkinson's Disease"),
            ],
          ),
          centerTitle: true,
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Parkinson’s Disease Overview',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 12),

                Text(
                  'Parkinson’s disease (PD) is a progressive neurological disorder that affects movement, muscle control, and balance. '
                  'It occurs when nerve cells in a region of the brain called the substantia nigra become damaged or die, reducing the production '
                  'of dopamine — a neurotransmitter essential for coordinated motion.',
                  style: TextStyle(fontSize: 16, height: 1.5),
                ),
                SizedBox(height: 20),

               
                Text(
                  'Common Symptoms',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  '• Tremors — rhythmic shaking, often starting in the hands or fingers\n'
                  '• Bradykinesia — slowness of movement that makes daily tasks difficult\n'
                  '• Muscle rigidity — stiffness and reduced range of motion\n'
                  '• Impaired posture and balance — difficulty standing or walking\n'
                  '• Speech changes — softer, slurred, or monotone speech patterns\n'
                  '• Facial masking — reduced expression or blinking\n'
                  '• Handwriting changes — smaller, cramped writing known as micrographia',
                  style: TextStyle(fontSize: 16, height: 1.5),
                ),
                SizedBox(height: 20),

               
                Text(
                  'Why Voice Analysis?',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  'Parkinson’s doesn’t just affect movement — it can also impact the vocal system. '
                  'As the muscles controlling speech weaken or stiffen, a person’s voice may become softer, slower, or less steady. '
                  'Subtle vocal biomarkers such as tremor frequency, breathiness, and pitch variation can reveal early neurological changes. '
                  'This makes voice analysis a promising, non-invasive tool for tracking disease progression.',
                  style: TextStyle(fontSize: 16, height: 1.5),
                ),
                SizedBox(height: 20),

                
                Text(
                  'How This App Relates',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  'The Parkinson’s Voice Analyzer allows users to record or upload short audio samples — such as a sustained “ahhhh” sound — '
                  'that can later be analyzed by machine learning models. These models, trained on research datasets, can identify vocal '
                  'patterns that may correlate with Parkinson’s symptoms. This helps researchers explore early detection methods and track '
                  'speech-related changes over time.',
                  style: TextStyle(fontSize: 16, height: 1.5),
                ),
                SizedBox(height: 20),

              
                Text(
                  'Research and Future Directions',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  'Recent studies have shown that AI-based models can achieve impressive accuracy when distinguishing Parkinson’s patients '
                  'from healthy individuals based on voice features like jitter, shimmer, and spectral entropy. By collecting and analyzing '
                  'voice data, researchers hope to make screening faster, more accessible, and more affordable globally — especially in '
                  'underserved regions.',
                  style: TextStyle(fontSize: 16, height: 1.5),
                ),
                SizedBox(height: 20),

                
                Text(
                  'Learn More',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text.rich(
                  TextSpan(
                    style: TextStyle(fontSize: 16, height: 1.5),
                    children: [
                      TextSpan(
                        text:
                            'For more information, you can visit the official Wikipedia article on Parkinson’s disease: ',
                      ),
                      TextSpan(
                        text: 'https://en.wikipedia.org/wiki/Parkinson%27s_disease',
                        style: TextStyle(
                          color: Colors.indigo,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 30),

                
                Divider(thickness: 1),
                SizedBox(height: 10),
                Text(
                  'Disclaimer: This application is for research and educational use only. '
                  'It is not a diagnostic tool and should not be used to make medical decisions. '
                  'If you suspect symptoms of Parkinson’s or any neurological condition, please consult a licensed healthcare professional.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    fontStyle: FontStyle.italic,
                    color: Colors.redAccent,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
}
