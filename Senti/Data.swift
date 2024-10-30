//
//  Data.swift
//  Senti
//
//  Created by Sammy Yousif on 10/11/24.
//

import SwiftData
import SwiftUI

enum Role: String, Codable {
    case assistant
    case user
    case system
}

@Model
class Message {
    var id: UUID = UUID()
    var role: Role = Role.user
    var content: String = ""
    var timestamp: Date = Date()
    
    @Relationship(inverse: \Thread.messages) var thread: Thread?
    
    init(role: Role, content: String, thread: Thread? = nil) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
        self.thread = thread
    }
}

@Model
class Thread {
    var id: UUID = UUID()
    var title: String? = nil
    var timestamp: Date = Date()
    
    @Relationship var messages: [Message]? = []
    
    var sortedMessages: [Message] {
        return messages?.sorted { $0.timestamp < $1.timestamp } ?? []
    }
    
    init() {
        self.id = UUID()
        self.timestamp = Date()
    }
}
