//
//  ContentView.swift
//  Senti
//
//  Created by Sammy Yousif on 10/10/24.
//

import SwiftUI
import AVFoundation
import Speech
import MLXLLM
import Accelerate

class PermissionManager: ObservableObject {
    @Published var micPermission: AVAudioSession.RecordPermission = .undetermined
    @Published var speechPermission: SFSpeechRecognizerAuthorizationStatus = .notDetermined
    
    func requestPermissions() {
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                self?.micPermission = granted ? .granted : .denied
            }
        }
        
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                self?.speechPermission = status
            }
        }
    }
    
    var allPermissionsGranted: Bool {
        return micPermission == .granted && speechPermission == .authorized
    }
}

class AudioManager: ObservableObject {
    @Published var audioLevel: CGFloat = 0.0
    @Published var currentSpeechText: String = ""
    @Published var recognizedSegments: [String] = []
    
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    
    var onPauseHandler: ((String) -> Void)?
    var onNewSpeechDetected: (() -> Void)?
    
    weak var speechQueue: SpeechQueue?
    
    private var isRecognizing: Bool = false
    
    func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        
        guard let audioEngine = audioEngine else { return }
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .allowBluetooth])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to set up audio session: \(error.localizedDescription)")
        }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, _) in
            self?.processAudioBuffer(buffer)
            if self?.isRecognizing == true {
                self?.recognitionRequest?.append(buffer)
            }
        }
        
        do {
            try audioEngine.start()
        } catch {
            print("Audio engine failed to start: \(error.localizedDescription)")
        }
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }

        let channelData0 = channelData[0]
        let length = vDSP_Length(buffer.frameLength)

        var rms: Float = 0.0
        vDSP_rmsqv(channelData0, 1, &rms, length)
        
        let decibels = 20 * log10(rms)
        
        let normalizedDecibels = (decibels + 40) / 40
        
        let scaledLevel = max(0, pow(normalizedDecibels, 0.6))
        
        DispatchQueue.main.async {
            self.audioLevel = CGFloat(scaledLevel)
        }
    }
    
    func startSpeechRecognition() {
        guard !isRecognizing else { return }
        
        isRecognizing = true
        setupSpeechRecognition()
    }
    
    private func setupSpeechRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        recognitionRequest?.shouldReportPartialResults = true
        
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] (result, error) in
            guard let self = self else { return }
            
            if let result = result {
                let bestTranscription = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.currentSpeechText = bestTranscription
                }
            }
        }
    }
    
    func stopRecognition() {
        isRecognizing = false
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
    }
    
    func resumeListening() {
        currentSpeechText = ""
        startSpeechRecognition()
    }
    
    func stopEverything() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        stopRecognition()
        audioEngine = nil
    }
}

struct AnimatedGradientCircle: View {
    @Binding var audioLevel: CGFloat
    @Binding var isFingerOnScreen: Bool
    @State private var gradientRotation: Double = 0
    @State private var currentScale: CGFloat = 0.5
    
    let gradient = LinearGradient(
        gradient: Gradient(colors: [.blue, .purple, .pink, .orange]),
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    
    var body: some View {
        Circle()
            .fill(gradient)
            .scaleEffect(currentScale)
            .animation(.spring(response: 0.3, dampingFraction: 0.6, blendDuration: 0), value: currentScale)
            .frame(width: 200, height: 200)
            .onChange(of: audioLevel) { newLevel in
                withAnimation {
                    currentScale = (isFingerOnScreen ? 0.5 : 0.6) + newLevel / 2
                }
            }
            .onChange(of: isFingerOnScreen) { _ in
                withAnimation {
                    currentScale = (isFingerOnScreen ? 0.5 : 0.6) + audioLevel / 2
                }
            }
    }
}

struct PermissionView: View {
    @ObservedObject var permissionManager: PermissionManager
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Permissions Required")
                .font(.title)
                .fontWeight(.bold)
            
            Text("This app requires microphone and speech recognition permissions to function properly.")
                .multilineTextAlignment(.center)
                .padding()
            
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: permissionManager.micPermission == .granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(permissionManager.micPermission == .granted ? .green : .red)
                    Text("Microphone Access")
                }
                
                HStack {
                    Image(systemName: permissionManager.speechPermission == .authorized ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(permissionManager.speechPermission == .authorized ? .green : .red)
                    Text("Speech Recognition")
                }
            }
            .padding()
            
            Button(action: {
                if permissionManager.micPermission == .denied || permissionManager.speechPermission == .denied {
                    if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(settingsUrl)
                    }
                } else {
                    permissionManager.requestPermissions()
                }
            }) {
                Text(permissionManager.micPermission == .denied || permissionManager.speechPermission == .denied ? "Open Settings" : "Enable Permissions")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
    }
}

struct SlidingText: View {
    let text: String
    @State private var opacity: Double = 0
    @State private var yOffset: CGFloat = 20
    
    var body: some View {
        Text(text)
            .opacity(opacity)
            .offset(y: yOffset)
            .onAppear {
                withAnimation(.easeOut(duration: 0.3)) {
                    opacity = 1
                    yOffset = 0
                }
            }
            .onDisappear {
                withAnimation(.easeIn(duration: 0.3)) {
                    opacity = 0
                    yOffset = -20
                }
            }
    }
}

struct FadingText: View {
    let text: String
    @State private var opacity: Double = 0
    
    var body: some View {
        Text(text)
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeIn(duration: 0.3)) {
                    opacity = 1
                }
            }
            .onDisappear {
                withAnimation(.easeOut(duration: 0.3)) {
                    opacity = 0
                }
            }
    }
}

struct ContentView: View {
    @Environment(\.modelContext) var modelContext
    @Environment(\.colorScheme) var colorScheme
    @StateObject private var audioManager = AudioManager()
    @StateObject private var permissionManager = PermissionManager()
    @State private var showPermissionSheet = false
    @EnvironmentObject var llm: LLMEvaluator
    @State var installed = true
    @State var currentThread: Thread?
    @State private var isFingerOnScreen = false
    @State private var isGenerating = false
    @State private var displayText: String = ""
    @State private var previousText: String = ""

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color(UIColor.systemBackground).edgesIgnoringSafeArea(.all)
                
                VStack(alignment: .center, spacing: 20) {
                    AnimatedGradientCircle(audioLevel: $audioManager.audioLevel, isFingerOnScreen: $isFingerOnScreen)
                    
                    if installed {
                        if isGenerating && !llm.isSpeaking {
                            FadingText(text: "Thinking")
                                .foregroundColor(.secondary)
                        } else if llm.isSpeaking {
                            SlidingText(text: llm.speechQueue.currentSentence)
                                .foregroundColor(.primary)
                                .multilineTextAlignment(.center)
                                .id(llm.speechQueue.currentSentence)
                                .padding(.horizontal)
                        } else if isFingerOnScreen && !audioManager.currentSpeechText.isEmpty {
                            FadingText(text: audioManager.currentSpeechText)
                                .foregroundColor(.primary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        } else if isFingerOnScreen {
                            FadingText(text: "Ask me anything")
                                .foregroundColor(.primary)
                                .multilineTextAlignment(.center)
                        } else {
                            FadingText(text: "Hold down to speak")
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("Downloading Model")
                            .foregroundColor(.primary)
                        ProgressView(value: llm.progress, total: 1)
                            .progressViewStyle(.linear)
                            .padding(.horizontal, 48)
                            .position(x: geometry.size.width / 2, y: geometry.size.height / 2 + 140)
                    }
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !isFingerOnScreen {
                            isFingerOnScreen = true
                            handleFingerDown()
                        }
                    }
                    .onEnded { _ in
                        isFingerOnScreen = false
                        handleFingerUp()
                    }
            )
        }
        .onAppear {
            let _ = isInstalled()
            checkAndRequestPermissions()
            setupAudioManagerHandler()
            connectAudioManagerToLLMEvaluator()
        }
        .onChange(of: permissionManager.allPermissionsGranted) { granted in
            if granted {
                showPermissionSheet = false
                audioManager.setupAudioEngine()
            } else {
                audioManager.stopEverything()
                updatePermissionSheetVisibility()
            }
        }
        .onChange(of: llm.progress) { _ in
            let _ = isInstalled()
        }
        .task {
            await loadLLM()
        }
        .sheet(isPresented: $showPermissionSheet, content: {
            PermissionView(permissionManager: permissionManager)
        })
        .ignoresSafeArea(.all)
    }
    
    private func handleFingerDown() {
        playHaptic()
        llm.cancelGeneration()
        audioManager.speechQueue?.cancelSpeech()
        audioManager.resumeListening()
    }
    
    private func handleFingerUp() {
        if !audioManager.currentSpeechText.isEmpty {
            generate(with: audioManager.currentSpeechText)
        }
        audioManager.stopRecognition()
    }
    
    private func checkAndRequestPermissions() {
        if permissionManager.allPermissionsGranted {
            audioManager.setupAudioEngine()
        } else if permissionManager.micPermission == .undetermined || permissionManager.speechPermission == .notDetermined {
            permissionManager.requestPermissions()
        } else {
            updatePermissionSheetVisibility()
        }
    }
    
    private func updatePermissionSheetVisibility() {
        showPermissionSheet = permissionManager.micPermission == .denied || permissionManager.speechPermission == .denied
    }
    
    private func setupAudioManagerHandler() {
        audioManager.onNewSpeechDetected = {
            llm.cancelGeneration()
            audioManager.speechQueue?.cancelSpeech()
            audioManager.resumeListening()
        }
    }
    
    private func generate(with segment: String) {
        if !segment.isEmpty {
            playHaptic()
            isGenerating = true
            Task {
                if currentThread == nil {
                    let newThread = Thread()
                    currentThread = newThread
                    await MainActor.run {
                        modelContext.insert(newThread)
                        try? modelContext.save()
                    }
                }
                
                if let currentThread = currentThread {
                    sendMessage(Message(role: .user, content: segment, thread: currentThread))
                    let output = await llm.generate(modelName: ModelConfiguration.defaultModel.name, systemPrompt: "You are helpful voice assistant. Respond using natural sounding dialogue.", thread: currentThread)
                    if !output.isEmpty {
                        sendMessage(Message(role: .assistant, content: output, thread: currentThread))
                    }
                }
                isGenerating = false
            }
        }
    }
    
    private func sendMessage(_ message: Message) {
        modelContext.insert(message)
        try? modelContext.save()
    }
    
    private func connectAudioManagerToLLMEvaluator() {
        audioManager.speechQueue = llm.speechQueue
        llm.setAudioManager(audioManager)
    }
    
    func playHaptic() {
        let impact = UIImpactFeedbackGenerator(style: .heavy)
        impact.impactOccurred()
    }
    
    func loadLLM() async {
        await llm.switchModel(.defaultModel)
    }
    
    func isInstalled() -> Bool {
        if llm.progress == 1 {
            installed = true
        }
        
        return installed && llm.progress == 1
    }
}

#Preview {
    ContentView()
}
