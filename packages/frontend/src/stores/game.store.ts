import { create } from 'zustand';

interface GameState {
  agentCount: number;
  incrementAgentCount: () => void;
}

export const useGameStore = create<GameState>((set) => ({
  agentCount: 0,
  incrementAgentCount: () => set((state) => ({ agentCount: state.agentCount + 1 })),
}));
