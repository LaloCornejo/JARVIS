const { ipcRenderer } = require("electron");

// Live2D Manager - Handles Live2D model loading and rendering
class Live2DManager {
  constructor(canvasId) {
    this.canvasId = canvasId;
    this.app = null;
    this.live2dModel = null;
    this.modelsPath = "../2dModels"; // Relative path to models directory
    // Mapping of display names to actual model file names
    this.modelMapping = {
      Sparkle: "Sparkle/Sparkle.model3.json",
      shiro: "shiro/Z.model3.json",
      shizuku: "shizuku/runtime/shizuku.model3.json",
      mao_pro: "mao_pro/runtime/mao_pro.model3.json",
      jean: "jean/简.model3.json",
    };
    this.availableModels = Object.keys(this.modelMapping);
    this.currentModel = null;
    this.isInitialized = false;
  }

  async initialize() {
    console.log("[DEBUG] Initializing Live2D manager...");
    console.log(`[DEBUG] Canvas ID: ${this.canvasId}`);
    console.log(`[DEBUG] Models path: ${this.modelsPath}`);
    console.log(`[DEBUG] Available models: ${this.availableModels.join(", ")}`);

    try {
      // Check if canvas exists
      const canvas = document.getElementById(this.canvasId);
      if (!canvas) {
        throw new Error(`Canvas with ID '${this.canvasId}' not found`);
      }
      console.log("[DEBUG] Canvas found successfully");

      // Initialize PixiJS application
      console.log("[DEBUG] Creating PIXI.Application...");
      this.app = new PIXI.Application({
        view: canvas,
        transparent: true,
        width: window.innerWidth,
        height: window.innerHeight,
      });
      console.log(
        `[DEBUG] App screen: ${this.app.screen.width}x${this.app.screen.height}`,
      );

      // Check if libraries are loaded
      console.log("[DEBUG] Checking PIXI availability:", typeof PIXI);
      console.log(
        "[DEBUG] Checking PIXI.live2d availability:",
        typeof PIXI.live2d,
      );
      console.log(
        "[DEBUG] Checking PIXI.live2d.Live2DModel availability:",
        typeof PIXI.live2d?.Live2DModel,
      );

      if (!PIXI.live2d) {
        throw new Error(
          "PIXI.live2d is not available. Make sure pixi-live2d-display is loaded correctly.",
        );
      }

      // Initialize Live2D plugin
      console.log("[DEBUG] Configuring Live2D plugin...");
      PIXI.live2d.config = {
        logLevel: PIXI.live2d.config.LOG_LEVEL_VERBOSE, // Enable verbose logging for debugging
      };
      console.log("[DEBUG] Live2D plugin configured");

      // Initially capture mouse events (disable passthrough)
      ipcRenderer.send("set-ignore-mouse-events", false);

      this.isInitialized = true;
      console.log("[DEBUG] Live2D manager initialized successfully");
    } catch (error) {
      console.error("[DEBUG] Failed to initialize Live2D manager:", error);
      throw error;
    }
  }

  async loadModel(modelName) {
    console.log(`[DEBUG] loadModel called with modelName: ${modelName}`);

    if (!this.isInitialized) {
      console.error("[DEBUG] Live2D manager not initialized");
      throw new Error("Live2D manager not initialized");
    }

    console.log(`[DEBUG] Available models: ${this.availableModels.join(", ")}`);

    if (!this.availableModels.includes(modelName)) {
      console.error(`[DEBUG] Model ${modelName} not in available models list`);
      throw new Error(
        `Model ${modelName} not available. Available models: ${this.availableModels.join(", ")}`,
      );
    }

    try {
      // Unload current model if exists
      if (this.live2dModel) {
        console.log("[DEBUG] Unloading current model");
        this.app.stage.removeChild(this.live2dModel);
        this.live2dModel.destroy();
      }

      // Construct model path using the mapping - use absolute path
      const modelFileName = this.modelMapping[modelName];
      const basePath = window.location.href.includes("file://")
        ? window.location.href.substring(
            0,
            window.location.href.lastIndexOf("/electron/"),
          )
        : ".";
      const modelPath = `${basePath}/2dModels/${modelFileName}`;

      console.log(`[DEBUG] Model mapping for ${modelName}: ${modelFileName}`);
      console.log(`[DEBUG] Base path: ${basePath}`);
      console.log(`[DEBUG] Full model path: ${modelPath}`);
      console.log(
        `[DEBUG] Attempting to load from: ${window.location.origin}/${modelPath}`,
      );

      // Load the Live2D model
      console.log(
        `[DEBUG] Calling PIXI.live2d.Live2DModel.from("${modelPath}")`,
      );

      // Debug: Check if path is accessible before loading
      try {
        const response = await fetch(modelPath);
        console.log(
          `[DEBUG] Path accessibility check - Status: ${response.status}, OK: ${response.ok}`,
        );
        if (!response.ok) {
          console.error(`[DEBUG] Path not accessible: ${modelPath}`);
          throw new Error(
            `Model file not accessible: ${modelPath} (HTTP ${response.status})`,
          );
        }
      } catch (fetchError) {
        console.error(
          `[DEBUG] Fetch check failed for path: ${modelPath}`,
          fetchError,
        );
      }

      this.live2dModel = await PIXI.live2d.Live2DModel.from(modelPath);
      console.log(`[DEBUG] Model loaded successfully!`);
      console.log(
        `[DEBUG] Model width: ${this.live2dModel.width}, height: ${this.live2dModel.height}`,
      );

      // Set model properties - small, bottom right
      const canvasWidth = this.app.screen.width;
      const canvasHeight = this.app.screen.height;

      const scale = (canvasHeight / 8500) * 0.8;
      this.live2dModel.scale.set(scale);
      this.live2dModel.anchor.set(0.5, 1); // anchor at bottom center
      this.live2dModel.x = canvasWidth - 150;
      this.live2dModel.y = canvasHeight + 400;
      console.log(
        `[DEBUG] Model scale: ${scale}, at: ${canvasWidth - 150}, ${canvasHeight - 20}`,
      );

      // Add to stage
      this.app.stage.addChild(this.live2dModel);

      // Enable interaction
      this.live2dModel.interactive = true;
      this.live2dModel.cursor = "default";

      // State variables
      this.isDragging = false;
      this.isHovering = false;
      this.isCtrlPressed = false;
      this.isShiftPressed = false;
      this.targetOpacity = 1;
      this.dragOffset = { x: 0, y: 0 };

      // Opacity animation
      this.app.ticker.add(() => {
        const lerpFactor = 0.05;
        this.live2dModel.alpha =
          this.live2dModel.alpha +
          (this.targetOpacity - this.live2dModel.alpha) * lerpFactor;
      });

      // Hover events - fade to transparent on hover
      this.live2dModel.on("pointerover", () => {
        this.isHovering = true;
        this.targetOpacity = 0;
        if (this.isCtrlPressed && this.isShiftPressed) {
          this.live2dModel.cursor = "grab";
        } else {
          this.live2dModel.cursor = "default";
        }
      });

      this.live2dModel.on("pointerout", () => {
        this.isHovering = false;
        this.targetOpacity = 1;
        this.live2dModel.cursor = "default";
      });

      // Dragging events
      this.live2dModel.on("pointerdown", (event) => {
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          this.isDragging = true;
          this.dragOffset.x = event.data.global.x - this.live2dModel.x;
          this.dragOffset.y = event.data.global.y - this.live2dModel.y;
          this.live2dModel.cursor = "grabbing";
        }
      });

      this.live2dModel.on("pointermove", (event) => {
        if (this.isDragging) {
          this.live2dModel.x = event.data.global.x - this.dragOffset.x;
          this.live2dModel.y = event.data.global.y - this.dragOffset.y;
        }
      });

      this.live2dModel.on("pointerup", () => {
        this.isDragging = false;
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          this.live2dModel.cursor = "grab";
        } else {
          this.live2dModel.cursor = "default";
        }
      });

      this.live2dModel.on("pointerupoutside", () => {
        this.isDragging = false;
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          this.live2dModel.cursor = "grab";
        } else {
          this.live2dModel.cursor = "default";
        }
      });

      // Key events for Ctrl + Shift (just for dragging, no passthrough)
      window.addEventListener("keydown", (event) => {
        if (event.key === "Control") this.isCtrlPressed = true;
        if (event.key === "Shift") this.isShiftPressed = true;
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          this.live2dModel.cursor = "grab";
        }
      });

      window.addEventListener("keyup", (event) => {
        if (event.key === "Control") this.isCtrlPressed = false;
        if (event.key === "Shift") this.isShiftPressed = false;
        if (!(this.isCtrlPressed && this.isShiftPressed)) {
          this.live2dModel.cursor = "default";
        }
      });

      // Zooming with scroll (only when Ctrl + Shift pressed and hovering)
      this.app.view.addEventListener("wheel", (event) => {
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          event.preventDefault();
          const zoomFactor = 0.02;
          const minScale = 0.05;
          const maxScale = 1.0;

          let newScale = this.live2dModel.scale.x;
          if (event.deltaY < 0) {
            newScale += zoomFactor;
          } else {
            newScale -= zoomFactor;
          }

          newScale = Math.max(minScale, Math.min(maxScale, newScale));
          this.live2dModel.scale.set(newScale);
        }
      });

      this.currentModel = modelName;
      console.log(`Successfully loaded Live2D model: ${modelName}`);

      // Start idle animation if available
      this.startIdleAnimation();
    } catch (error) {
      console.error(`[DEBUG] Failed to load Live2D model ${modelName}:`, error);
      console.error(`[DEBUG] Error details:`, {
        modelName,
        modelPath: `${this.modelsPath}/${this.modelMapping[modelName]}`,
        errorMessage: error.message,
        errorStack: error.stack,
      });

      // Try to provide helpful debugging information
      if (
        error.message.includes("404") ||
        error.message.includes("Not Found")
      ) {
        console.error(
          `[DEBUG] Model file not found. Check if the file exists at the expected path.`,
        );
      } else if (error.message.includes("CORS")) {
        console.error(
          `[DEBUG] CORS error detected. Check if the model files are served with proper CORS headers.`,
        );
      } else if (error.message.includes("JSON")) {
        console.error(
          `[DEBUG] JSON parsing error. Check if the .model3.json file is valid JSON.`,
        );
      }

      throw error;
    }
  }

  startIdleAnimation() {
    if (!this.live2dModel) return;

    try {
      const motions = this.live2dModel.internalModel?.motionManager?.motions;
      const expressions =
        this.live2dModel.internalModel?.motionManager?.expressionManager
          ?.expressions;

      console.log(
        "[Live2D] Available motion groups:",
        motions ? Object.keys(motions) : "none",
      );
      console.log(
        "[Live2D] Available expressions:",
        expressions ? Object.keys(expressions) : "none",
      );

      if (motions && motions.Idle && motions.Idle.length > 0) {
        console.log("[Live2D] Starting Idle motion");
        this.live2dModel.internalModel.motionManager.startMotion("Idle", 0, 2);
      } else if (motions && Object.keys(motions).length > 0) {
        const firstMotionGroup = Object.keys(motions)[0];
        console.log(
          `[Live2D] No Idle found, playing first motion: ${firstMotionGroup}`,
        );
        this.live2dModel.internalModel.motionManager.startMotion(
          firstMotionGroup,
          0,
          2,
        );
      } else {
        console.log("[Live2D] No motions found in model");
      }

      if (this.live2dModel.internalModel.motionManager.eyeBlink) {
        try {
          this.live2dModel.internalModel.motionManager.eyeBlink.enable();
          console.log("[Live2D] Eye blinking enabled");
        } catch (e) {}
      }

      if (this.live2dModel.internalModel.physics) {
        try {
          this.live2dModel.internalModel.physics.enable();
          console.log("[Live2D] Physics enabled");
        } catch (e) {}
      }

      // Start random idle expression cycling
      this.startIdleExpressionCycle(expressions);
    } catch (error) {
      console.log("[Live2D] Could not start idle animation:", error);
    }
  }

  startIdleExpressionCycle(expressions) {
    if (!expressions || Object.keys(expressions).length === 0) return;

    const expressionKeys = Object.keys(expressions);
    const neutralExpressions = expressionKeys.filter(
      (k) =>
        !k.includes("生气") &&
        !k.includes("泪") &&
        !k.includes("血") &&
        !k.includes("脸黑"),
    );

    if (neutralExpressions.length === 0) return;

    // Clear any existing interval
    if (this.idleExpressionInterval) {
      clearInterval(this.idleExpressionInterval);
    }

    // Random expression every 3-8 seconds
    const cycleExpression = () => {
      try {
        const randomExp =
          neutralExpressions[
            Math.floor(Math.random() * neutralExpressions.length)
          ];
        const expressionManager =
          this.live2dModel.internalModel.motionManager.expressionManager;
        expressionManager.setExpression(randomExp);
        console.log(`[Live2D] Idle expression: ${randomExp}`);

        // Return to default after 2-4 seconds
        const returnTimeout = setTimeout(
          () => {
            try {
              expressionManager.setExpression("Idle") ||
                expressionManager.setExpression("neutral") ||
                expressionManager.setExpression(expressionKeys[0]);
            } catch (e) {}
          },
          2000 + Math.random() * 2000,
        );
      } catch (e) {}

      const nextCycle = 3000 + Math.random() * 5000;
      this.idleExpressionTimeout = setTimeout(cycleExpression, nextCycle);
    };

    // Start first cycle after 5 seconds
    this.idleExpressionTimeout = setTimeout(cycleExpression, 5000);
  }

  stopIdleExpressionCycle() {
    if (this.idleExpressionInterval) {
      clearInterval(this.idleExpressionInterval);
      this.idleExpressionInterval = null;
    }
    if (this.idleExpressionTimeout) {
      clearTimeout(this.idleExpressionTimeout);
      this.idleExpressionTimeout = null;
    }
  }

  playMotion(motionName, priority = 1) {
    if (!this.live2dModel) return;

    try {
      const motionManager = this.live2dModel.internalModel.motionManager;
      const motions = motionManager?.motions;

      if (!motions) {
        console.log(`[Live2D] No motions available`);
        return;
      }

      // Try exact match first, then case-insensitive search
      let targetMotion = motionName;
      if (!motions[motionName]) {
        const keys = Object.keys(motions);
        targetMotion = keys.find(
          (k) => k.toLowerCase() === motionName.toLowerCase(),
        );
      }

      if (targetMotion && motions[targetMotion]) {
        console.log(`[Live2D] Playing motion: ${targetMotion}`);
        motionManager.startMotion(targetMotion, 0, priority);
      } else {
        console.log(
          `[Live2D] Motion "${motionName}" not found. Available: ${Object.keys(motions).join(", ")}`,
        );
      }
    } catch (error) {
      console.log(`[Live2D] Could not play motion ${motionName}:`, error);
    }
  }

  setExpression(expressionName) {
    if (!this.live2dModel) return;

    // Stop idle expression cycle when manual expression is set
    this.stopIdleExpressionCycle();

    try {
      const expressionManager =
        this.live2dModel.internalModel?.motionManager?.expressionManager;
      const expressions = expressionManager?.expressions;

      if (!expressions) {
        console.log(`[Live2D] No expressions available`);
        return;
      }

      let targetExpression = expressionName;
      if (!expressions[expressionName]) {
        const keys = Object.keys(expressions);
        targetExpression = keys.find(
          (k) => k.toLowerCase() === expressionName.toLowerCase(),
        );
      }

      if (targetExpression && expressions[targetExpression]) {
        console.log(`[Live2D] Setting expression: ${targetExpression}`);
        expressionManager.setExpression(targetExpression);
      } else {
        console.log(
          `[Live2D] Expression "${expressionName}" not found. Available: ${Object.keys(expressions).join(", ")}`,
        );
      }

      // Resume idle cycle after 3 seconds
      setTimeout(() => {
        this.startIdleExpressionCycle(expressions);
      }, 3000);
    } catch (error) {
      console.log(
        `[Live2D] Could not set expression ${expressionName}:`,
        error,
      );
    }
  }

  setParameter(parameterId, value) {
    if (!this.live2dModel) return;

    try {
      this.live2dModel.internalModel.setParameterValueById(parameterId, value);
    } catch (error) {
      console.log(`Could not set parameter ${parameterId}:`, error);
    }
  }

  // Handle window resize
  onResize() {
    if (this.app && this.live2dModel) {
      this.live2dModel.position.set(
        this.app.screen.width / 2,
        this.app.screen.height / 2,
      );
    }
  }

  // Cleanup
  destroy() {
    if (this.live2dModel) {
      this.live2dModel.destroy();
      this.live2dModel = null;
    }
    if (this.app) {
      this.app.destroy();
      this.app = null;
    }
    this.isInitialized = false;
  }

  // Get available models list
  getAvailableModels() {
    return this.availableModels;
  }

  // Get current model name
  getCurrentModel() {
    return this.currentModel;
  }
}

// Export the class for use in other modules
export { Live2DManager };
