import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { FileProcessor } from './pages/FileProcessor';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';
import { History } from './pages/History';
import { ResultsViewer } from './pages/ResultsViewer';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/process" element={<FileProcessor />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/results/:resultId" element={<ResultsViewer />} />
          <Route path="/results/session/:sessionId" element={<ResultsViewer />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;