const { ipcRenderer } = require("electron");

// JARVIS with Live2D integration
class JarvisWithLive2D {
  constructor() {
    this.ws = null;
    this.live2dManager = null;
    this.responseBox = document.getElementById("response-box");
    this.responseTimeout = null;
    this.currentResponse = "";
    if (this.responseBox) {
      this.responseBox.classList.add("visible");
    }
    this.initializeLive2D();
    setTimeout(() => this.connect(), 5000);
  }

  // Safe helper to access responseBox
  getResponseBox() {
    if (!this.responseBox) {
      this.responseBox = document.getElementById("response-box");
    }
    return this.responseBox;
  }

  connect() {
    this.ws = new WebSocket("ws://localhost:8000/ws");

    this.ws.onopen = () => {
      console.log("Connected to JARVIS server with Live2D");
      const box = this.getResponseBox();
      if (box) {
        box.textContent = "Connected to JARVIS";
        box.classList.add("visible");
      }
    };

    this.ws.onclose = () => {
      console.log("Disconnected from JARVIS server");
      const box = this.getResponseBox();
      if (box) {
        box.textContent = "Disconnected from JARVIS";
      }
      setTimeout(() => this.connect(), 5000);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      const box = this.getResponseBox();
      if (box) {
        box.textContent = "Connection failed";
      }
    };

    this.ws.onmessage = (event) => {
      try {
        this.handleMessage(JSON.parse(event.data));
      } catch (e) {
        const box = this.getResponseBox();
        if (box) {
          box.textContent = `Parse error: ${e.message}`;
        }
      }
    };
  }

  async initializeLive2D() {
    console.log("[DEBUG] Starting Live2D initialization...");

    try {
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
    }
  }

  handleScreenDimensions(dimensions) {
    console.log("Screen dimensions:", dimensions);
    this.screenWidth = dimensions.width;
    this.screenHeight = dimensions.height;

    if (this.live2dManager) {
      this.live2dManager.onResize();
    }
  }

  handleMessage(data) {
    const { type, content, source, username, chat_id, channel_id } = data;

    switch (type) {
      case 'user_message':
        this.currentResponse = '';
        if (source) {
          const sourceLabel = this.getSourceLabel(source, username, chat_id, channel_id);
          const box = this.getResponseBox();
          if (box) {
            box.innerHTML = `<span class="source-label">${sourceLabel}</span>`;
          }
        }
        this.triggerLive2DReaction("talk");
        break;
      case 'streaming_chunk':
        this.currentResponse += content;
        this.showResponse(this.currentResponse, source);
        if (Math.random() < 0.15) {
          this.triggerLive2DReaction("talk");
        }
        break;
      case 'assistant_message':
        this.currentResponse = content;
        this.showResponse(content, source);
        this.handleAssistantMessage(content);
        break;
      case 'message_complete':
        this.handleMessageComplete();
        break;
      default:
        const box2 = this.getResponseBox();
        if (box2) {
          box2.textContent = `Received: ${type}`;
        }
    }
  }

  getSourceLabel(source, username, chatId, channelId) {
    const user = username || 'Unknown';
    switch (source) {
      case 'telegram':
        return `[Telegram: ${user}] `;
      case 'discord':
        return `[Discord: ${user}] `;
      case 'whatsapp':
        return `[WhatsApp: ${user}] `;
      case 'tui':
        return `[TUI] `;
      default:
        return `[${source || 'Unknown'}: ${user}] `;
    }
  }

  handleAssistantMessage(content) {
    this.currentResponse = content;
    this.showResponse(content);

    const lowerContent = content.toLowerCase();

    if (lowerContent.includes("hello") || lowerContent.includes("hi") || lowerContent.includes("greetings")) {
      this.triggerLive2DReaction("happy");
    } else if (lowerContent.includes("error") || lowerContent.includes("failed") || lowerContent.includes("sorry")) {
      this.triggerLive2DReaction("sad");
    } else if (lowerContent.includes("yes") || lowerContent.includes("correct") || lowerContent.includes("good")) {
      this.triggerLive2DReaction("happy");
    } else if (lowerContent.includes("no") || lowerContent.includes("wrong") || lowerContent.includes("bad")) {
      this.triggerLive2DReaction("angry");
    } else {
      this.triggerLive2DReaction("talk");
    }
  }

  handleStreamingChunk(content) {
    this.currentResponse += content;
    this.showResponse(this.currentResponse);

    if (Math.random() < 0.1) {
      this.triggerLive2DReaction("idle");
    }
  }

  handleMessageComplete() {
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

  showResponse(content, source = null) {
    const box = this.getResponseBox();
    if (!box) return;

    if (content.length > 2000) {
      content = content.substring(0, 2000) + '...';
    }
    const html = marked.parse ? marked.parse(content) : marked(content);
    
    if (source && source !== 'tui') {
      box.innerHTML = `<span class="source-${source}">${html}</span>`;
    } else {
      box.innerHTML = html;
    }
    this.applyBionic(box);
    box.style.fontSize = content.length > 500 ? '13px' : '16px';
    if (content.length > 1000) {
      box.style.maxHeight = '400px';
    } else {
      box.style.maxHeight = '200px';
    }
    box.classList.add('visible');
    if (this.responseTimeout) {
      clearTimeout(this.responseTimeout);
    }
    this.responseTimeout = setTimeout(() => {
      box.classList.remove('visible');
    }, 60000);
  }

  handleLive2DCommand(command) {
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
          if (parameter && parameter.id !== undefined && parameter.value !== undefined) {
            this.live2dManager.setParameter(parameter.id, parameter.value);
          }
          break;
      }
    } catch (error) {
      console.error("Error handling Live2D command:", error);
    }
  }

  triggerLive2DReaction(reactionType) {
    if (!this.live2dManager || !this.live2dManager.live2dModel) return;

    switch (reactionType) {
      case "talk":
        this.live2dManager.playMotion("talk", 1);
        break;
      case "idle":
        this.live2dManager.startIdleAnimation();
        break;
      case "happy":
        this.tryExpressions(["happy", "exp_01", "开心", "爱心眼", "脸红"]);
        break;
      case "angry":
        this.tryExpressions(["angry", "生气", "脸黑", "血"]);
        break;
      case "sad":
        this.tryExpressions(["sad", "泪", "白眼"]);
        break;
      case "surprised":
        this.tryExpressions(["surprised", "星星眼"]);
        break;
      case "blink":
        this.live2dManager.playMotion("blink", 2);
        break;
    }
  }

  tryExpressions(expressionNames) {
    for (const name of expressionNames) {
      try {
        const expressionManager = this.live2dManager.live2dModel.internalModel?.motionManager?.expressionManager;
        const expressions = expressionManager?.expressions;
        
        if (expressions && expressions[name]) {
          expressionManager.setExpression(name);
          console.log(`[Live2D] Applied expression: ${name}`);
          return true;
        }
      } catch (e) {}
    }
    console.log(`[Live2D] No matching expression found for: ${expressionNames.join(", ")}`);
    return false;
  }

  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "user_message", content: message }));
      return true;
    }
    return false;
  }

  getAvailableModels() {
    return this.live2dManager ? this.live2dManager.getAvailableModels() : [];
  }

  getCurrentModel() {
    return this.live2dManager ? this.live2dManager.getCurrentModel() : null;
  }

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

document.addEventListener("DOMContentLoaded", () => {
  const jarvis = new JarvisWithLive2D();

  // Enable passthrough after a short delay to ensure window is ready
  setTimeout(() => {
    ipcRenderer.send("set-ignore-mouse-events", true);
  }, 1000);

  ipcRenderer.on("screen-dimensions", (event, dimensions) => {
    jarvis.handleScreenDimensions(dimensions);
  });

  // Global mouse tracking for model following
  let followEnabled = true;
  let targetX = 0, targetY = 0;
  
  ipcRenderer.on("mouse-position", (event, pos) => {
    if (!jarvis.live2dManager || !jarvis.live2dManager.live2dModel) return;
    
    const model = jarvis.live2dManager.live2dModel;
    const app = jarvis.live2dManager.app;
    if (!app) return;
    
    // Calculate mouse position relative to window
    const relX = pos.x - pos.windowX;
    const relY = pos.y - pos.windowY;
    
    // Clamp to window bounds (0-1 range)
    const normX = Math.max(0, Math.min(1, relX / pos.windowWidth));
    const normY = Math.max(0, Math.min(1, relY / pos.windowHeight));
    
    console.log(`[Mouse] rel: ${relX.toFixed(0)},${relY.toFixed(0)} norm: ${normX.toFixed(2)},${normY.toFixed(2)}`);
    
    targetX = normX;
    targetY = normY;
    
    // Try different methods to set look-at
    try {
      const internal = model.internalModel;
      if (internal) {
        // Method 1: Check for lookAt in motionManager
        const mm = internal.motionManager;
        if (mm) {
          if (mm.lookAt) {
            mm.lookAt.angle = (normX - 0.5) * 30;
            mm.lookAt.angleY = (normY - 0.5) * 15;
          }
          // Method 2: Try setting parameter directly
          if (mm.coreModel) {
            const ParamAngleX = internal.model.settings?.find(p => p.id === 'ParamAngleX');
            const ParamAngleY = internal.model.settings?.find(p => p.id === 'ParamAngleY');
            if (ParamAngleX) {
              mm.coreModel.setParameterValueById('ParamAngleX', (normX - 0.5) * 30);
            }
            if (ParamAngleY) {
              mm.coreModel.setParameterValueById('ParamAngleY', (normY - 0.5) * 15);
            }
          }
        }
        
        // Method 3: Try View translation
        if (internal.model?.transform) {
          const view = internal.model.transform;
          const offsetX = (normX - 0.5) * 0.1;
          const offsetY = (normY - 0.5) * 0.1;
          view.position.x = offsetX;
          view.position.y = offsetY;
        }
      }
    } catch (e) {
      // Silently fail - tracking is optional
    }
  });

  // Listen for global toggle events from main process
  ipcRenderer.on("toggle-follow", () => {
    followEnabled = !followEnabled;
    console.log("Follow mode:", followEnabled);
    if (toggleBtn) {
      toggleBtn.textContent = followEnabled ? "Follow: ON" : "Passthrough: OFF";
      toggleBtn.style.background = followEnabled ? "rgba(0,100,200,0.5)" : "rgba(0,0,0,0.5)";
    }
  });

  ipcRenderer.on("toggle-passthrough", () => {
    updatePassthrough(!passthroughEnabled);
  });

  window.addEventListener("resize", () => {
    if (jarvis.live2dManager) {
      jarvis.live2dManager.onResize();
    }
  });

  window.jarvis = jarvis;

  window.testLive2D = {
    switchModel: (modelName) => jarvis.switchModel(modelName),
    getAvailableModels: () => jarvis.getAvailableModels(),
    getCurrentModel: () => jarvis.getCurrentModel(),
    playMotion: (motionName) => {
      if (jarvis.live2dManager) {
        jarvis.live2dManager.playMotion(motionName);
      }
    },
    setExpression: (expressionName) => {
      if (jarvis.live2dManager) {
        jarvis.live2dManager.setExpression(expressionName);
      }
    },
    triggerReaction: (reactionType) => jarvis.triggerLive2DReaction(reactionType),
  };
});
