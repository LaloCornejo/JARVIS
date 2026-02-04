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

      // Initialize PixiJS application (PIXI.js v6 API)
      console.log("[DEBUG] Creating PIXI.Application...");
      this.app = new PIXI.Application({
        view: canvas,
        transparent: true, // Transparent background
        resizeTo: window,
      });
      console.log("[DEBUG] PIXI.Application created successfully");

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

      // Initially allow mouse passthrough
      ipcRenderer.send("set-ignore-mouse-events", true);

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

      // Construct model path using the mapping
      const modelFileName = this.modelMapping[modelName];
      const modelPath = `${this.modelsPath}/${modelFileName}`;

      console.log(`[DEBUG] Model mapping for ${modelName}: ${modelFileName}`);
      console.log(`[DEBUG] Constructed model path: ${modelPath}`);
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
        // Continue anyway, as Live2D might handle the loading differently
      }

      this.live2dModel = await PIXI.live2d.Live2DModel.from(modelPath);
      console.log(`[DEBUG] Model loaded successfully!`);

      // Set model properties
      this.live2dModel.scale.set(0.08); // Scale down the model
      this.live2dModel.anchor.set(-2.2, -0.2); // Center the model
      this.live2dModel.position.set(
        this.app.screen.width / 2,
        this.app.screen.height / 2,
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
      this.targetOpacity = 0;
      this.dragOffset = { x: 0, y: 0 };

      // Opacity animation
      this.app.ticker.add(() => {
        const lerpFactor = 0.05;
        this.live2dModel.alpha =
          this.live2dModel.alpha +
          (this.targetOpacity - this.live2dModel.alpha) * lerpFactor;
      });

      // Hover events
      this.live2dModel.on("pointerover", () => {
        this.isHovering = true;
        if (this.isCtrlPressed && this.isShiftPressed) {
          this.live2dModel.cursor = "grab";
          this.targetOpacity = 1;
          ipcRenderer.send("set-ignore-mouse-events", false);
        } else {
          ipcRenderer.send("set-ignore-mouse-events", true);
        }
      });

      this.live2dModel.on("pointerout", () => {
        this.isHovering = false;
        this.live2dModel.cursor = "default";
        this.targetOpacity = 0;
        ipcRenderer.send("set-ignore-mouse-events", true);
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

      // Key events for Ctrl + Shift
      window.addEventListener("keydown", (event) => {
        if (event.key === "Control") this.isCtrlPressed = true;
        if (event.key === "Shift") this.isShiftPressed = true;
        if (this.isCtrlPressed && this.isShiftPressed && this.isHovering) {
          this.live2dModel.cursor = "grab";
          this.targetOpacity = 1;
          ipcRenderer.send("set-ignore-mouse-events", false);
        }
      });

      window.addEventListener("keyup", (event) => {
        if (event.key === "Control") this.isCtrlPressed = false;
        if (event.key === "Shift") this.isShiftPressed = false;
        if (!(this.isCtrlPressed && this.isShiftPressed)) {
          this.live2dModel.cursor = "default";
          this.targetOpacity = 0;
          ipcRenderer.send("set-ignore-mouse-events", true);
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
      // Try to play an idle motion (looping)
      const motions = this.live2dModel.internalModel.motionManager.motions;
      if (motions && motions.idle && motions.idle.length > 0) {
        this.live2dModel.startMotion("idle", 0, 1);
      } else if (motions && Object.keys(motions).length > 0) {
        // Play first available motion (looping)
        const firstMotionGroup = Object.keys(motions)[0];
        this.live2dModel.startMotion(firstMotionGroup, 0, 1);
      } else {
        console.log("No motions found in model");
      }
    } catch (error) {
      console.log("Could not start idle animation:", error);
    }
  }

  playMotion(motionName, priority = 1) {
    if (!this.live2dModel) return;

    try {
      this.live2dModel.motion(motionName, priority);
    } catch (error) {
      console.log(`Could not play motion ${motionName}:`, error);
    }
  }

  setExpression(expressionName) {
    if (!this.live2dModel) return;

    try {
      this.live2dModel.expression(expressionName);
    } catch (error) {
      console.log(`Could not set expression ${expressionName}:`, error);
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
