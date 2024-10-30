//
//  LLMEvaluator.swift
//  Senti
//
//  Created by Sammy Yousif on 10/10/24.
//

import MLX
import MLXLLM
import MLXRandom
import SwiftUI
import AVFoundation
import Speech
import Accelerate

@Observable
@MainActor
class LLMEvaluator: ObservableObject {
    var running = false
    var output = ""
    var modelInfo = ""
    var stat = ""
    var progress = 0.0

    var modelConfiguration = ModelConfiguration.defaultModel
    
    let streamSentenceSplitter = StreamSentenceSplitter()
    let speechQueue = SpeechQueue()
    weak var audioManager: AudioManager?
    
    private var currentGenerationId: UUID?
    private var currentGenerationTask: Task<String, Error>?
    
    var isSpeaking = false
    
    init() {
        streamSentenceSplitter.sentenceHandler = { [weak self] sentence in
            self?.speechQueue.enqueue(sentence: sentence)
        }
        
        speechQueue.onSpeechStarted = { [weak self] in
            self?.isSpeaking = true
            self?.toggleIdleTimer(disable: true)
        }
                
        speechQueue.onSpeechFinished = { [weak self] in
            self?.isSpeaking = false
            self?.toggleIdleTimer(disable: false)
            self?.audioManager?.resumeListening()
        }
    }

    func switchModel(_ model: ModelConfiguration) async {
        progress = 0.0 // reset progress
        loadState = .idle
        modelConfiguration = model
        _ = try? await load(modelName: model.name)
    }

    let generateParameters = GenerateParameters(temperature: 0.5)
    let maxTokens = 4096
    let displayEveryNTokens = 4

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    func load(modelName: String) async throws -> ModelContainer {
        let model = getModelByName(modelName)
        
        switch loadState {
        case .idle:
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await MLXLLM.loadModelContainer(configuration: model!)
            {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                    self.progress = progress.fractionCompleted
                }
            }
            self.modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(MLX.GPU.activeMemory / 1024 / 1024)M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    func generate(modelName: String, systemPrompt: String, thread: Thread) async -> String {
        guard !running else { return "" }

        running = true
        self.output = ""
        
        let generationId = UUID()
        self.currentGenerationId = generationId
        
        var text = ""
        
        toggleIdleTimer(disable: true)

        currentGenerationTask = Task { [weak self] in
            guard let self = self else { return "" }
            do {
                let modelContainer = try await self.load(modelName: modelName)
                let extraEOSTokens = self.modelConfiguration.extraEOSTokens

                let promptHistory = self.modelConfiguration.getPromptHistory(thread: thread, systemPrompt: systemPrompt)
                let prompt = self.modelConfiguration.prepare(prompt: promptHistory)

                let promptTokens = await modelContainer.perform { _, tokenizer in
                    tokenizer.encode(text: prompt)
                }

                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

                let result = await modelContainer.perform { model, tokenizer in
                    MLXLLM.generate(
                        promptTokens: promptTokens, parameters: self.generateParameters, model: model,
                        tokenizer: tokenizer, extraEOSTokens: extraEOSTokens
                    ) { tokens in
                        if Task.isCancelled { return .stop }
                        
                        if tokens.count % self.displayEveryNTokens == 0 {
                            let newText = tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                let newChunk = String(newText.suffix(from: text.endIndex))
                                self.output = newText
                                if self.currentGenerationId == generationId {
                                    print(newChunk)
                                    self.streamSentenceSplitter.handleStreamChunk(newChunk)
                                }
                                text = newText
                            }
                        }

                        if tokens.count >= self.maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                }

                if result.output != self.output {
                    let newChunk = String(result.output.suffix(from: text.endIndex))
                    self.output = result.output
                    if self.currentGenerationId == generationId, !newChunk.isEmpty {
                        self.streamSentenceSplitter.handleStreamChunk(newChunk)
                    }
                }
                self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

                // Wait for speech to finish before returning
                while self.isSpeaking {
                    try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
                }

                return self.output
            } catch {
                return "Failed: \(error)"
            }
        }

        let result = (try? await currentGenerationTask?.value) ?? ""
        running = false
        toggleIdleTimer(disable: false)
        return result
    }
    
    func cancelGeneration() {
        output = ""
        currentGenerationTask?.cancel()
        currentGenerationId = nil
        streamSentenceSplitter.clear() 
        speechQueue.cancelSpeech()
        running = false
        isSpeaking = false
        toggleIdleTimer(disable: false)
    }
    
    private func toggleIdleTimer(disable: Bool) {
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = disable
        }
    }
    
    func getModelByName(_ name: String) -> ModelConfiguration? {
        if let model = ModelConfiguration.availableModels.first(where: { $0.name == name }) {
            return model
        } else {
            return nil
        }
    }
    
    func setAudioManager(_ audioManager: AudioManager) {
        self.audioManager = audioManager
        self.speechQueue.audioManager = audioManager
    }
}

class StreamSentenceSplitter {
    private var currentSentence: String = ""
    private let sentenceTerminators: [Character] = [".", "!", "?"]
    private let abbreviations: Set<String> = ["st", "mr", "mrs", "dr", "ms", "jr", "sr"]
    private let punctuatedWords: Set<String> = [
        "u.s", "u.s.a", "u.k", "e.g", "i.e", "etc", "ph.d", "a.m", "p.m",
        "b.c", "a.d", "fig", "bros", "dept", "corp", "inc", "co", "vs",
        "gen", "sen", "rev", "hon", "gov", "lt", "cmdr", "approx", "est",
        "alt", "def", "n.b", "p.s", "r.s.v.p", "tel", "temp", "vet", "viz"
    ]
    private var lastCharacterWasTerminator = false
    private var lastCharacterWasNewline = false
    
    var sentenceHandler: ((String) -> Void)?
    
    func handleStreamChunk(_ newValue: String) {
        for character in newValue {
            if character.isNewline {
                if !currentSentence.isEmpty {
                    // Send the current sentence if we have one
                    if let handler = sentenceHandler {
                        handler(currentSentence)
                    }
                    currentSentence = ""
                }
                lastCharacterWasNewline = true
                lastCharacterWasTerminator = false
                continue
            }
            
            if lastCharacterWasNewline {
                lastCharacterWasNewline = false
                currentSentence = String(character)
                continue
            }
            
            if lastCharacterWasTerminator {
                if character.isWhitespace {
                    // Check if the last word is a punctuated word
                    if !isPunctuatedWord() {
                        if let handler = sentenceHandler {
                            handler(currentSentence)
                        }
                        currentSentence = String(character)
                    } else {
                        currentSentence.append(character)
                    }
                } else {
                    currentSentence.append(character)
                }
                lastCharacterWasTerminator = false
            } else {
                if sentenceTerminators.contains(character) {
                    if canTerminateSentence(with: character) {
                        currentSentence.append(character)
                        lastCharacterWasTerminator = true
                    } else {
                        currentSentence.append(character)
                    }
                } else {
                    currentSentence.append(character)
                }
            }
        }
    }
    
    func clear() {
        currentSentence = ""
        lastCharacterWasTerminator = false
        lastCharacterWasNewline = false
    }
    
    private func isPunctuatedWord() -> Bool {
        let words = currentSentence.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: " ")
            .map(String.init)
        
        guard let lastWord = words.last else { return false }
        
        // Remove any trailing punctuation for comparison
        let cleanedWord = lastWord.trimmingCharacters(in: .punctuationCharacters).lowercased()
        
        // Check if it matches known punctuated words
        if punctuatedWords.contains(cleanedWord) {
            return true
        }
        
        // Check for capitalized letter patterns with periods (e.g., U.S., A.I.)
        let components = lastWord.components(separatedBy: ".")
        if components.count > 1 {
            // Check if each component is a single capital letter
            let isAcronym = components.allSatisfy { component in
                component.count == 1 && component.rangeOfCharacter(from: .uppercaseLetters) != nil
            }
            if isAcronym {
                return true
            }
        }
        
        return false
    }
    
    private func canTerminateSentence(with terminator: Character) -> Bool {
        let trimmedSentence = currentSentence.trimmingCharacters(in: .whitespacesAndNewlines)
        if let lastWord = trimmedSentence.split(separator: " ").last {
            let lastWordString = String(lastWord).lowercased()
            if abbreviations.contains(lastWordString) {
                return false
            }
            if lastWordString.count == 1 {
                return false
            }
            if let lastCharacter = lastWordString.last, lastCharacter.isNumber {
                return false
            }
        }
        return true
    }
}

class SpeechQueue: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    static let speechSynthesizer = AVSpeechSynthesizer()
    
    private var sentenceQueue: [String] = []
    @Published var currentSentence: String = ""
    @Published var currentWord: String = ""
    @Published var isSpeaking: Bool = false
    
    var onSpeechStarted: (() -> Void)?
    var onSpeechFinished: (() -> Void)?
    weak var audioManager: AudioManager?
    
    private var isProcessingSpeech: Bool = false
    
    override init() {
        super.init()
        Self.speechSynthesizer.delegate = self
        
        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .duckOthers])
            try AVAudioSession.sharedInstance().setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to set up audio session for SpeechQueue: \(error.localizedDescription)")
        }
    }
    
    func cancelSpeech() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.sentenceQueue.removeAll()
            Self.speechSynthesizer.stopSpeaking(at: .immediate)
            self.isSpeaking = false
            self.isProcessingSpeech = false
            self.onSpeechFinished?()
        }
    }
    
    func enqueue(sentence: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.sentenceQueue.append(sentence)
            self.processSpeechQueue()
        }
    }
    
    private func processSpeechQueue() {
        guard !isProcessingSpeech else { return }
        
        isProcessingSpeech = true
        speakNextSentence()
    }
    
    private func speakNextSentence() {
        guard !sentenceQueue.isEmpty else {
            isSpeaking = false
            isProcessingSpeech = false
            onSpeechFinished?()
            return
        }
        
        let sentence = sentenceQueue.removeFirst()
        print("speaking:", sentence)
        
        currentSentence = sentence
        let utterance = AVSpeechUtterance(string: sentence)
        utterance.rate = utterance.rate * 1.1
        
        isSpeaking = true
        onSpeechStarted?()
        Self.speechSynthesizer.speak(utterance)
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { [weak self] in
            self?.isSpeaking = true
            self?.onSpeechStarted?()
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { [weak self] in
            self?.speakNextSentence()
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, willSpeakRangeOfSpeechString characterRange: NSRange, utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { [weak self] in
            let sentence = utterance.speechString
            if let range = Range(characterRange, in: sentence) {
                self?.currentWord = String(sentence[range])
            }
        }
    }
}
