import { useEffect, useRef } from 'react';
import Phaser from 'phaser';
import { MainScene } from './game/MainScene';

function App() {
  const gameContainerRef = useRef<HTMLDivElement>(null);
  const gameRef = useRef<Phaser.Game | null>(null);

  useEffect(() => {
    if (gameContainerRef.current && !gameRef.current) {
      const config: Phaser.Types.Core.GameConfig = {
        type: Phaser.AUTO,
        parent: gameContainerRef.current,
        width: 800,
        height: 600,
        backgroundColor: '#242424',
        scene: [MainScene],
      };

      gameRef.current = new Phaser.Game(config);
    }

    return () => {
      if (gameRef.current) {
        gameRef.current.destroy(true);
        gameRef.current = null;
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white">
      <header className="mb-4">
        <h1 className="text-4xl font-bold text-green-500">Agent Crossing</h1>
      </header>
      <div ref={gameContainerRef} className="rounded-lg shadow-2xl overflow-hidden border-4 border-gray-700" />
      <footer className="mt-4 text-gray-400">
        <p>Use React for UI overlays, Phaser for the world.</p>
      </footer>
    </div>
  );
}

export default App;
