const { ipcRenderer } = require("electron");

// JARVIS with Live2D integration
class JarvisWithLive2D {
  constructor() {
    this.ws = null;
    this.live2dManager = null;
    this.responseBox = document.getElementById("response-box");
    this.responseTimeout = null;
    this.currentResponse = "";
    this.responseBox.classList.add("visible");
    this.initializeLive2D();
    // Delay connect to allow server to start
    setTimeout(() => this.connect(), 5000);
  }

  connect() {
    this.ws = new WebSocket("ws://localhost:8000/ws");

    this.ws.onopen = () => {
      console.log("Connected to JARVIS server with Live2D");
      this.responseBox.textContent = "Connected to JARVIS";
      this.responseBox.classList.add("visible");
    };

    this.ws.onclose = () => {
      console.log("Disconnected from JARVIS server");
      this.responseBox.textContent = "Disconnected from JARVIS";
      // Attempt to reconnect after 5 seconds
      setTimeout(() => this.connect(), 5000);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.responseBox.textContent = "Connection failed";
    };

    this.ws.onmessage = (event) => {
      try {
        this.handleMessage(JSON.parse(event.data));
      } catch (e) {
        this.responseBox.textContent = `Parse error: ${e.message}`;
      }
    };
  }

  async initializeLive2D() {
    console.log("[DEBUG] Starting Live2D initialization...");

    try {
      // Import Live2D manager
      console.log("[DEBUG] Importing Live2D manager...");
      const { Live2DManager } = await import("./live2d-manager.js");
      console.log("[DEBUG] Live2D manager imported successfully");

      console.log("[DEBUG] Creating Live2D manager instance...");
      this.live2dManager = new Live2DManager("live2d-canvas");

      console.log("[DEBUG] Initializing Live2D manager...");
      await this.live2dManager.initialize();

      console.log("[DEBUG] Loading default model (jean)...");
      await this.live2dManager.loadModel("jean");

      console.log("[DEBUG] Live2D initialized successfully");
    } catch (error) {
      console.error("[DEBUG] Failed to initialize Live2D:", error);
      console.error("[DEBUG] Error details:", {
        message: error.message,
        stack: error.stack,
        name: error.name,
      });
    }
  }

  // Handle screen dimensions from main process
  handleScreenDimensions(dimensions) {
    console.log("Screen dimensions:", dimensions);
    this.screenWidth = dimensions.width;
    this.screenHeight = dimensions.height;

    // Update Live2D canvas size if manager exists
    if (this.live2dManager) {
      this.live2dManager.onResize();
    }
  }

  handleMessage(data) {
    const { type, content } = data;

    switch (type) {
      case 'user_message':
        this.currentResponse = '';
        break;
      case 'streaming_chunk':
        this.currentResponse += content;
        this.showResponse(this.currentResponse);
        break;
      case 'assistant_message':
        this.currentResponse = content;
        this.showResponse(content);
        break;
      case 'message_complete':
        // Keep the final response
        break;
      default:
        this.responseBox.textContent = `Received: ${type}`;
    }
  }

  handleAssistantMessage(content) {
    this.currentResponse = content;
    // Display the response
    this.showResponse(content);

    // Analyze message content for appropriate reaction
    const lowerContent = content.toLowerCase();

    if (
      lowerContent.includes("hello") ||
      lowerContent.includes("hi") ||
      lowerContent.includes("greetings")
    ) {
      this.triggerLive2DReaction("happy");
    } else if (
      lowerContent.includes("error") ||
      lowerContent.includes("failed") ||
      lowerContent.includes("sorry")
    ) {
      this.triggerLive2DReaction("sad");
    } else if (
      lowerContent.includes("yes") ||
      lowerContent.includes("correct") ||
      lowerContent.includes("good")
    ) {
      this.triggerLive2DReaction("happy");
    } else if (
      lowerContent.includes("no") ||
      lowerContent.includes("wrong") ||
      lowerContent.includes("bad")
    ) {
      this.triggerLive2DReaction("angry");
    } else {
      // Default reaction for regular messages
      this.triggerLive2DReaction("talk");
    }
  }

  handleStreamingChunk(content) {
    this.currentResponse += content;
    this.showResponse(this.currentResponse);

    // Subtle reaction during streaming
    if (Math.random() < 0.1) {
      // 10% chance to react
      this.triggerLive2DReaction("idle");
    }
  }

  handleMessageComplete() {
    // Reset to idle state
    this.triggerLive2DReaction("idle");
  }

  applyBionic(element) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);
    let node;
    while (node = walker.nextNode()) {
      const text = node.textContent;
      const newText = text.replace(/\b\w+/g, word => {
        if (word.length <= 1) return word;
        const boldLen = Math.max(1, Math.ceil(word.length / 2));
        return `<strong>${word.slice(0, boldLen)}</strong>${word.slice(boldLen)}`;
      });
      const span = document.createElement('span');
      span.innerHTML = newText;
      node.parentNode.replaceChild(span, node);
    }
  }

  showResponse(content) {
    // Truncate if needed
    if (content.length > 2000) {
      content = content.substring(0, 2000) + '...';
    }
    // Render Markdown
    const html = marked.parse ? marked.parse(content) : marked(content);
    this.responseBox.innerHTML = html;
    // Apply bionic reading
    this.applyBionic(this.responseBox);
    // Dynamic font size
    this.responseBox.style.fontSize = content.length > 500 ? '13px' : '16px';
    // Adjust height for long responses
    if (content.length > 1000) {
      this.responseBox.style.maxHeight = '400px';
    } else {
      this.responseBox.style.maxHeight = '200px';
    }
    this.responseBox.classList.add('visible');
    if (this.responseTimeout) {
      clearTimeout(this.responseTimeout);
    }
    this.responseTimeout = setTimeout(() => {
      this.responseBox.classList.remove('visible');
    }, 60000); // Hide after 1 minute
  }

  handleLive2DCommand(command) {
    // Handle specific Live2D commands from server
    if (!this.live2dManager) return;

    try {
      const { action, model, motion, expression, parameter } = command;

      switch (action) {
        case "load_model":
          if (model) {
            this.live2dManager.loadModel(model);
          }
          break;
        case "play_motion":
          if (motion) {
            this.live2dManager.playMotion(motion);
          }
          break;
        case "set_expression":
          if (expression) {
            this.live2dManager.setExpression(expression);
          }
          break;
        case "set_parameter":
          if (
            parameter &&
            parameter.id !== undefined &&
            parameter.value !== undefined
          ) {
            this.live2dManager.setParameter(parameter.id, parameter.value);
          }
          break;
      }
    } catch (error) {
      console.error("Error handling Live2D command:", error);
    }
  }

  triggerLive2DReaction(reactionType) {
    if (!this.live2dManager) return;

    switch (reactionType) {
      case "talk":
        // Try to play a talking motion
        this.live2dManager.playMotion("talk");
        break;
      case "idle":
        // Return to idle state
        this.live2dManager.startIdleAnimation();
        break;
      case "happy":
        // Try different expressions for happy
        this.live2dManager.setExpression("happy") ||
          this.live2dManager.setExpression("exp_01") ||
          this.live2dManager.setExpression("开心") ||
          this.live2dManager.setExpression("爱心眼");
        break;
      case "angry":
        // Try different expressions for angry
        this.live2dManager.setExpression("angry") ||
          this.live2dManager.setExpression("生气") ||
          this.live2dManager.setExpression("脸黑");
        break;
      case "sad":
        // Try different expressions for sad
        this.live2dManager.setExpression("sad") ||
          this.live2dManager.setExpression("泪") ||
          this.live2dManager.setExpression("白眼");
        break;
      case "surprised":
        // Try different expressions for surprised
        this.live2dManager.setExpression("surprised") ||
          this.live2dManager.setExpression("星星眼");
        break;
      case "blink":
        // Trigger eye blink
        this.live2dManager.playMotion("blink");
        break;
    }
  }

  // Method to send messages programmatically
  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "user_message",
          content: message,
        }),
      );
      return true;
    }
    return false;
  }

  // Get available Live2D models
  getAvailableModels() {
    return this.live2dManager ? this.live2dManager.getAvailableModels() : [];
  }

  // Get current model
  getCurrentModel() {
    return this.live2dManager ? this.live2dManager.getCurrentModel() : null;
  }

  // Switch to different model
  async switchModel(modelName) {
    if (this.live2dManager) {
      try {
        await this.live2dManager.loadModel(modelName);
        return true;
      } catch (error) {
        console.error("Failed to switch model:", error);
        return false;
      }
    }
    return false;
  }
}

// Initialize JARVIS with Live2D when the page loads
document.addEventListener("DOMContentLoaded", () => {
  const jarvis = new JarvisWithLive2D();

  // Listen for screen dimensions from main process
  ipcRenderer.on("screen-dimensions", (event, dimensions) => {
    jarvis.handleScreenDimensions(dimensions);
  });

  // Handle window resize
  window.addEventListener("resize", () => {
    if (jarvis.live2dManager) {
      jarvis.live2dManager.onResize();
    }
  });

  // Make jarvis instance globally available for debugging
  window.jarvis = jarvis;

  // Add debug functions for testing
  window.testLive2D = {
    switchModel: (modelName) => {
      console.log(`Switching to model: ${modelName}`);
      return jarvis.switchModel(modelName);
    },
    getAvailableModels: () => jarvis.getAvailableModels(),
    getCurrentModel: () => jarvis.getCurrentModel(),
    playMotion: (motionName) => {
      console.log(`Playing motion: ${motionName}`);
      if (jarvis.live2dManager) {
        jarvis.live2dManager.playMotion(motionName);
      }
    },
    setExpression: (expressionName) => {
      console.log(`Setting expression: ${expressionName}`);
      if (jarvis.live2dManager) {
        jarvis.live2dManager.setExpression(expressionName);
      }
    },
    triggerReaction: (reactionType) => {
      console.log(`Triggering reaction: ${reactionType}`);
      jarvis.triggerLive2DReaction(reactionType);
    },
    // Add additional debug functions
    testFilePaths: () => {
      console.log("[DEBUG] Testing file paths...");
      const testPaths = [
        "../2dModels/jean/简.model3.json",
        "../2dModels/shiro/Z.model3.json",
        "../2dModels/shizuku/runtime/shizuku.model3.json",
        "../2dModels/mao_pro/runtime/mao_pro.model3.json",
        "../2dModels/Sparkle/Sparkle.model3.json",
      ];

      testPaths.forEach(async (path) => {
        try {
          const response = await fetch(path);
          console.log(
            `[DEBUG] Path ${path}: Status ${response.status} (${response.ok ? "OK" : "FAILED"})`,
          );
        } catch (error) {
          console.log(`[DEBUG] Path ${path}: Error - ${error.message}`);
        }
      });
    },
    debugManager: () => {
      console.log("[DEBUG] Live2D Manager Debug Info:");
      if (jarvis.live2dManager) {
        console.log("Manager exists:", jarvis.live2dManager);
        console.log("Is initialized:", jarvis.live2dManager.isInitialized);
        console.log("Current model:", jarvis.live2dManager.currentModel);
        console.log("Models path:", jarvis.live2dManager.modelsPath);
        console.log("Available models:", jarvis.live2dManager.availableModels);
        console.log("PIXI app exists:", !!jarvis.live2dManager.app);
        console.log("Live2D model exists:", !!jarvis.live2dManager.live2dModel);
      } else {
        console.log("Live2D manager not found");
      }
    },
  };

  console.log("Debug functions available via window.testLive2D");
  console.log("Available models:", window.testLive2D.getAvailableModels());
  console.log("Additional debug commands:");
  console.log(
    "- window.testLive2D.testFilePaths() - Test if model files are accessible",
  );
  console.log(
    "- window.testLive2D.debugManager() - Show manager internal state",
  );
});

