const axios = require("axios");
const tiktoken = require("@dqbd/tiktoken");

class LLMTalker {
  constructor({
    promptString = "No matter what I say, you always reply 'KOKOKAKA I DON't KNOW'.",
    messages = [],
    apiKey = "",
    apiUrl = "https://api.openai.com/v1/chat/completions",
    model = "gpt-4o",
    maxMessages = 30,
    maxTokens = 32000, // 32k is enough, no need to go 128k.
  }) {
    // system prompt is included in the messages.
    if (!apiKey) {
      throw new Error("API key is required.");
    }
    this.promptString = promptString;
    this.messages = [...messages];
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.model = model;
    this.encoder = tiktoken.encoding_for_model(model);
    this.maxMessages = maxMessages;
    this.maxTokens = maxTokens;
  }

  static removeAllSystemPrompts() {
    // remove the prompt from the beginning of the messages.
    // use while if there are multiple prompts. For most cases, there should be only one prompt.
    while (this.messages.length > 0 && this.messages[0].role === "system") {
      this.messages.shift();
    }
  }

  static createMessage({ messageString, role }) {
    if (!messageString || !role) {
      throw new Error("Both messageString and role are required to create a message.");
    }
    return { role, content: messageString };
  }

  calculateToken(message) {
    let tokenCount = 0;

    const roleTokens = this.encoder.encode(message.role); // Encode the role (e.g., "user", "assistant")
    const contentTokens = this.encoder.encode(message.content); // Encode the content
    tokenCount += roleTokens.length + contentTokens.length + 4; // +4 for message format overhead

    return tokenCount;
  }

  getMessagesWithinTokenLimit({ topPromptMessages = [], addTopPromptMessages = true, tokenLimitDeduction = 0, messageLimitDeduction = 0 }) {
    // keep up to "this.maxMessages" messages, and up to "this.maxTokens" tokens.
    const activeTokenLimit = this.maxTokens - tokenLimitDeduction;
    const activeMessageLimit = this.maxMessages - messageLimitDeduction;
    let totalTokens = 0;
    let totalMessages = 0;
    let messages = [];

    for (let i = topPromptMessages.length - 1; i >= 0; i--) {
      const message = topPromptMessages[i];
      const tokens = this.calculateToken(message);
      if (totalTokens + tokens <= activeTokenLimit && totalMessages < activeMessageLimit) {
        // add the top prompt later after all body messages are added.
        totalTokens += tokens;
        totalMessages++;
      } else {
        break;
      }
    }

    for (let i = this.messages.length - 1; i >= 0; i--) {
      const message = this.messages[i];
      const tokens = this.calculateToken(message);
      if (totalTokens + tokens <= activeTokenLimit && totalMessages < activeMessageLimit) {
        messages.unshift(message);
        totalTokens += tokens;
        totalMessages++;
      } else {
        break;
      }
    }
    // add prompt to the beginning of the messages.
    // messages.unshift(createMessage({ messageString: this.promptString, role: "system" }));
    // add toopPromptMessages to the beginning of the messages.
    if (addTopPromptMessages) {
      messages = topPromptMessages.concat(messages);
    }
    return messages;
  }

  /**
   * Send a message to the GPT model and receive a response.
   * @param {Object} options - Options for sending the message.
   * @param {string} options.messageString - The user message to send.
   * @returns {Promise<Array>} - The updated list of messages.
   */
  async sendMessage({ newMessage = null, requirePreprocessing = true }) {
    try {
      if (newMessage) {
        // this is in case of the new message is not already in the messages.
        this.messages.push(newMessage);
      }
      let sendingMessages = this.messages;
      console.log("requirePreprocessing: ", requirePreprocessing);
      if (requirePreprocessing) {
        console.log("now preprocessing");
        sendingMessages = this.getMessagesWithinTokenLimit({
          topPromptMessages: [
            LLMTalker.createMessage({ messageString: this.promptString, role: "system" }),
          ],
        });
      }
      console.log("sendingMessages: ", sendingMessages);

      // Make the API request
      const response = await axios.post(
        this.apiUrl,
        {
          model: this.model,
          messages: sendingMessages,
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
          },
        }
      );

      const reply = response.data.choices[0].message.content;
      const assistantMessage = LLMTalker.createMessage({ messageString: reply, role: "assistant" });
      this.messages.push(assistantMessage);

      return assistantMessage;
    } catch (error) {
      console.error("Error communicating with GPT:", error.response?.data || error.message);
      throw new Error("Failed to communicate with GPT model.");
    }
  }
}

module.exports = LLMTalker;