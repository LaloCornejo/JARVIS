const { app, BrowserWindow, screen, ipcMain } = require("electron");
const { spawn } = require('child_process');
const net = require('net');
const path = require("path");

let robot;
try {
  robot = require('robotjs');
} catch (e) {
  console.log("robotjs not available");
}

let mainWindow;
let mouseInterval;

function isPortOpen(host, port) {
  return new Promise((resolve) => {
    const socket = net.createConnection(port, host, () => {
      socket.end();
      resolve(true);
    });
    socket.on('error', () => resolve(false));
    socket.setTimeout(1000, () => {
      socket.destroy();
      resolve(false);
    });
  });
}

function createWindow() {
  const displays = screen.getAllDisplays();
  const display = displays.length > 1 ? displays[1] : screen.getPrimaryDisplay();
  const workArea = display.workArea;
  const windowWidth = workArea.width;
  const windowHeight = workArea.height;
  const windowX = workArea.x;
  const windowY = workArea.y;
  console.log(`Display ${display.id} work area: ${windowWidth}x${windowHeight} at (${windowX},${windowY})`);

  mainWindow = new BrowserWindow({
    x: windowX,
    y: windowY,
    width: windowWidth,
    height: windowHeight,
    transparent: true,
    frame: false,
    backgroundColor: '#00000000',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    alwaysOnTop: true,
    skipTaskbar: false,
    show: true,
    resizable: false,
    movable: false,
    focusable: true,
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.setBounds({
      x: windowX,
      y: windowY,
      width: windowWidth,
      height: windowHeight
    });
  });

  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow.setIgnoreMouseEvents(true);
  });

  mainWindow.loadFile("index.html");

  // Send screen dimensions to renderer when it's ready
  mainWindow.webContents.once("dom-ready", () => {
    console.log("DOM ready, sending screen dimensions...");
    mainWindow.webContents.send("screen-dimensions", {
      width: windowWidth,
      height: windowHeight,
    });
  });

  mainWindow.setHasShadow(false);

   mainWindow.setFocusable(true);

   // Track global mouse position and send to renderer
   try {
     if (!robot) {
       console.log("Using PowerShell for mouse tracking");
       // Use PowerShell to get mouse position
       const getMousePos = () => {
         const { execSync } = require('child_process');
         try {
           const result = execSync('powershell -Command "[System.Windows.Forms.Cursor]::Position.X; [System.Windows.Forms.Cursor]::Position.Y"', { encoding: 'utf8' });
           const [x, y] = result.trim().split('\n').map(Number);
           return { x, y };
         } catch (e) {
           return { x: 0, y: 0 };
         }
       };
       
       mouseInterval = setInterval(() => {
         if (mainWindow && !mainWindow.isDestroyed()) {
           const mousePos = getMousePos();
           const windowBounds = mainWindow.getBounds();
           mainWindow.webContents.send("mouse-position", {
             x: mousePos.x,
             y: mousePos.y,
             windowX: windowBounds.x,
             windowY: windowBounds.y,
             windowWidth: windowBounds.width,
             windowHeight: windowBounds.height,
           });
         }
       }, 50);
     } else {
       mouseInterval = setInterval(() => {
         if (mainWindow && !mainWindow.isDestroyed()) {
           const mousePos = robot.getMousePos();
           const windowBounds = mainWindow.getBounds();
           mainWindow.webContents.send("mouse-position", {
             x: mousePos.x,
             y: mousePos.y,
             windowX: windowBounds.x,
             windowY: windowBounds.y,
             windowWidth: windowBounds.width,
             windowHeight: windowBounds.height,
           });
         }
       }, 50);
     }
   } catch (err) {
     console.error("Mouse tracking error:", err);
   } // 20fps for smooth following

   mainWindow.on("closed", () => {
     mainWindow = null;
     if (mouseInterval) {
       clearInterval(mouseInterval);
     }
   });

     // Listen for ignore mouse events from renderer
    ipcMain.on('set-ignore-mouse-events', (event, ignore) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        // Use forward option so clicks pass through but button still works
        mainWindow.setIgnoreMouseEvents(ignore, { forward: true });
      }
    });
  }

app.whenReady().then(async () => {
  const portOpen = await isPortOpen('localhost', 8000);
  if (!portOpen) {
    console.log('Starting JARVIS server...');
    const serverProcess = spawn('uv', ['run', 'python', '-c', 'from core.websocket_server import run_server; run_server()'], { cwd: path.join(__dirname, '..'), stdio: 'inherit' });
  } else {
    console.log('JARVIS server already running');
  }

  createWindow();
});

app.on("window-all-closed", () => {
  // Unregister shortcuts
  globalShortcut.unregisterAll();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
