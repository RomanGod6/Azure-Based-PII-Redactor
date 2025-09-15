/// <reference types="react-scripts" />

declare global {
  interface Window {
    electronAPI: {
      getBackendPort: () => Promise<number>;
      checkBackendHealth: () => Promise<boolean>;
      showOpenDialog: (options: any) => Promise<any>;
      showSaveDialog: (options: any) => Promise<any>;
      showMessageBox: (options: any) => Promise<any>;
      platform: string;
      versions: {
        node: string;
        chrome: string;
        electron: string;
      };
    };
  }
}