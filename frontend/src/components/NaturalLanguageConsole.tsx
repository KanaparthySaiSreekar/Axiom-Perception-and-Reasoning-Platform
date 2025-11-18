'use client';

import { useState, useRef, useEffect } from 'react';
import { apiClient } from '@/services/api';
import type { StructuredCommand } from '@/types';

interface Message {
  id: string;
  type: 'user' | 'system' | 'llm' | 'error';
  content: string;
  timestamp: Date;
  command?: StructuredCommand;
}

export function NaturalLanguageConsole() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      type: 'system',
      content: 'AI Command Console initialized. Type natural language commands below.',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [pendingCommand, setPendingCommand] = useState<StructuredCommand | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);

    try {
      // Process natural language command
      const command = await apiClient.processNaturalLanguageCommand(input);

      const llmMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'llm',
        content: command.reasoning,
        timestamp: new Date(),
        command,
      };

      setMessages((prev) => [...prev, llmMessage]);
      setPendingCommand(command);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExecuteCommand = async () => {
    if (!pendingCommand) return;

    try {
      await apiClient.executeAction(pendingCommand.action);

      const systemMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: `Executing action: ${pendingCommand.action.action_type}. Estimated duration: ${pendingCommand.estimated_duration_sec}s`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, systemMessage]);
      setPendingCommand(null);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'error',
        content: `Execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleCancelCommand = () => {
    setPendingCommand(null);

    const systemMessage: Message = {
      id: Date.now().toString(),
      type: 'system',
      content: 'Command cancelled by user.',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, systemMessage]);
  };

  return (
    <div className="flex flex-col h-96">
      {/* Console Output */}
      <div className="console flex-1 overflow-y-auto mb-4 max-h-80">
        {messages.map((message) => (
          <div key={message.id} className="mb-3">
            <div className="text-xs text-gray-500 mb-1">
              [{message.timestamp.toLocaleTimeString()}]
            </div>

            {message.type === 'user' && (
              <div className="console-prompt">
                &gt; {message.content}
              </div>
            )}

            {message.type === 'system' && (
              <div className="console-output text-blue-400">
                {message.content}
              </div>
            )}

            {message.type === 'llm' && message.command && (
              <div className="space-y-2">
                <div className="console-output text-green-400">
                  💡 {message.content}
                </div>

                <div className="bg-gray-900 p-3 rounded border border-gray-700">
                  <div className="text-sm mb-2">
                    <span className="text-yellow-400">Structured Command:</span>
                  </div>
                  <div className="text-xs space-y-1">
                    <div>
                      <span className="text-gray-400">Action:</span>{' '}
                      <span className="text-white">{message.command.action.action_type}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Parameters:</span>{' '}
                      <pre className="text-white mt-1">
                        {JSON.stringify(message.command.action.parameters, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <span className="text-gray-400">Safety:</span>{' '}
                      <span
                        className={
                          message.command.safety_validated ? 'text-green-400' : 'text-red-400'
                        }
                      >
                        {message.command.safety_validated ? '✓ Validated' : '✗ Failed'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Estimated Duration:</span>{' '}
                      <span className="text-white">
                        {message.command.estimated_duration_sec}s
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {message.type === 'error' && (
              <div className="console-error">
                ❌ {message.content}
              </div>
            )}
          </div>
        ))}

        {isProcessing && (
          <div className="console-output text-yellow-400 animate-pulse">
            🤖 Processing command with LLM...
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Pending Command Actions */}
      {pendingCommand && (
        <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-600 rounded">
          <div className="text-sm font-semibold text-yellow-400 mb-2">
            ⚠️ Command Ready for Execution
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleExecuteCommand}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm font-medium"
            >
              Execute Command
            </button>
            <button
              onClick={handleCancelCommand}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter natural language command..."
          className="flex-1 px-3 py-2 bg-black border border-gray-700 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          disabled={isProcessing}
        />
        <button
          type="submit"
          disabled={isProcessing || !input.trim()}
          className="px-6 py-2 bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-primary-foreground rounded font-medium transition-colors"
        >
          {isProcessing ? 'Processing...' : 'Send'}
        </button>
      </form>

      {/* Quick Commands */}
      <div className="mt-3 flex gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">Quick commands:</span>
        {['Pick up the bottle', 'Pour 200ml into the cup', 'Go to home position'].map(
          (cmd) => (
            <button
              key={cmd}
              onClick={() => setInput(cmd)}
              className="text-xs px-2 py-1 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded"
            >
              {cmd}
            </button>
          )
        )}
      </div>
    </div>
  );
}
