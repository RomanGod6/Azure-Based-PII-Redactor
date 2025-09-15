const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Load .env file
try {
  require('dotenv').config();
} catch (error) {
  console.log('No .env file found or dotenv not installed');
}

const isDev = process.env.NODE_ENV !== 'production';

console.log('Electron Environment:', {
  NODE_ENV: process.env.NODE_ENV,
  isDev: isDev,
  PORT: process.env.PORT
});

let mainWindow;
let backendProcess;
let backendPort = 8080;

// Backend health check
const checkBackendHealth = () => {
  return new Promise((resolve) => {
    const http = require('http');
    const req = http.get(`http://localhost:${backendPort}/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => resolve(false));
  });
};

// Start Go backend server
const startBackend = () => {
  return new Promise(async (resolve, reject) => {
    if (isDev) {
      // In development mode, just check if backend is already running
      console.log('Development mode: checking if backend is already running...');
      const isHealthy = await checkBackendHealth();
      if (isHealthy) {
        console.log('✅ Backend already running at http://localhost:' + backendPort);
        resolve();
        return;
      } else {
        reject(new Error('Backend not running. Please start it with "npm run dev:backend"'));
        return;
      }
    }
    
    // Production mode - start the compiled binary
    const backendPath = path.join(process.resourcesPath, 'backend');
    const backendExe = process.platform === 'win32' ? 'redactify-server.exe' : 'redactify-server';
    
    console.log('Starting backend:', backendExe, 'from', backendPath);
    
    backendProcess = spawn(backendExe, [], {
      cwd: backendPath,
      env: {
        ...process.env,
        PORT: backendPort.toString(),
        NODE_ENV: 'production'
      },
      stdio: 'pipe'
    });

    backendProcess.stdout.on('data', (data) => {
      console.log('Backend stdout:', data.toString());
    });

    backendProcess.stderr.on('data', (data) => {
      console.log('Backend stderr:', data.toString());
    });

    backendProcess.on('close', (code) => {
      console.log('Backend process exited with code', code);
      if (!app.isQuitting) {
        // Restart backend if it crashes unexpectedly
        setTimeout(startBackend, 2000);
      }
    });

    // Wait for backend to be ready
    const checkHealth = async () => {
      for (let i = 0; i < 30; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        if (await checkBackendHealth()) {
          resolve();
          return;
        }
      }
      reject(new Error('Backend failed to start'));
    };

    await checkHealth();
  });
};

// Create the main application window
const createMainWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: !isDev // Disable web security in development
    },
    icon: path.join(__dirname, 'frontend', 'public', 'icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false
  });

  // Load the frontend
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, 'frontend/build/index.html')}`;
  
  console.log('Loading URL:', startUrl);
  mainWindow.loadURL(startUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
};

// App event handlers
app.whenReady().then(async () => {
  try {
    await startBackend();
    createMainWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox('Startup Error', 'Failed to start the backend server. Please try again.');
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  app.isQuitting = true;
  
  // Kill backend process
  if (backendProcess) {
    backendProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  app.isQuitting = true;
  if (backendProcess) {
    backendProcess.kill();
  }
});

// IPC handlers
ipcMain.handle('get-backend-port', () => backendPort);

ipcMain.handle('check-backend-health', async () => {
  return await checkBackendHealth();
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-message-box', async (event, options) => {
  const result = await dialog.showMessageBox(mainWindow, options);
  return result;
});

// Health check interval
setInterval(async () => {
  if (!await checkBackendHealth() && !app.isQuitting) {
    console.log('Backend health check failed, attempting restart...');
    if (backendProcess) {
      backendProcess.kill();
    }
    try {
      await startBackend();
    } catch (error) {
      console.error('Failed to restart backend:', error);
    }
  }
}, 30000); // Check every 30 seconds