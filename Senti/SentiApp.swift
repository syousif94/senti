//
//  SentiApp.swift
//  Senti
//
//  Created by Sammy Yousif on 10/10/24.
//

import SwiftUI
import SwiftData

@main
struct SentiApp: App {
    @StateObject var llm = LLMEvaluator()
    
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Thread.self,
            Message.self
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .environment(llm)
        .modelContainer(sharedModelContainer)
    }
}
