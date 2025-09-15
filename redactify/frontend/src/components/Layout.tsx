import React from 'react';
import { Navigation } from './Navigation';
import { StatusBanner } from './StatusBanner';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="flex h-screen bg-gray-50">
      <StatusBanner />
      <Navigation />
      <main className="flex-1 overflow-auto">
        <div className="p-6 pt-20"> {/* Add top padding for status banner */}
          {children}
        </div>
      </main>
    </div>
  );
};