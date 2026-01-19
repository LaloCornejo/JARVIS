const { app, BrowserWindow, screen, globalShortcut, ipcMain } = require("electron");
const { spawn } = require('child_process');
const net = require('net');
const path = require("path");

let mainWindow;

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
  // Get the primary display's dimensions
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  console.log(`Screen dimensions: ${width}x${height}`);

  mainWindow = new BrowserWindow({
    width: width,
    height: height,
    transparent: true,
    frame: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    alwaysOnTop: true,
    skipTaskbar: false,
    show: true,
    movable: false,
    resizable: false,
    minimizable: true,
    maximizable: false,
    closable: true,
    focusable: false,
  });

   mainWindow.loadFile("index.html");

  // Send screen dimensions to renderer when it's ready
  mainWindow.webContents.once("dom-ready", () => {
    console.log("DOM ready, sending screen dimensions...");
    mainWindow.webContents.send("screen-dimensions", {
      width: screen.getPrimaryDisplay().workAreaSize.width,
      height: screen.getPrimaryDisplay().workAreaSize.height,
    });
  });

  // Make window completely invisible and non-interactive
  // mainWindow.setIgnoreMouseEvents(true);
  mainWindow.setHasShadow(false);

   mainWindow.setFocusable(false);

   mainWindow.on("closed", () => {
     mainWindow = null;
   });

   // Listen for ignore mouse events from renderer
   ipcMain.on('set-ignore-mouse-events', (event, ignore) => {
     if (mainWindow && !mainWindow.isDestroyed()) {
       mainWindow.setIgnoreMouseEvents(ignore);
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

  // Register global shortcut to open DevTools
  const registered1 = globalShortcut.register(
    "CommandOrControl+Shift+I",
    () => {
      console.log("Opening DevTools via Ctrl+Shift+I");
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.openDevTools({ mode: "detach" });
      }
    },
  );

  const registered2 = globalShortcut.register("F12", () => {
    console.log("Opening DevTools via F12");
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
  });

  console.log("DevTools shortcuts registered:", registered1, registered2);
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
