export interface AgentPersona {
  id: string;
  name: string;
  age: number;
  description: string;
  traits: string[];
  schedule: ScheduleItem[];
}

export interface ScheduleItem {
  time: string;
  activity: string;
  location: string;
}

export interface AgentState {
  id: string;
  position: { x: number; y: number };
  currentActivity: string;
  currentLocation: string;
  affinity: Record<string, number>;
}

export interface MemoryEntry {
  id: string;
  agentId: string;
  content: string;
  timestamp: string;
  type: 'observation' | 'interaction' | 'reflection';
  metadata?: Record<string, any>;
}
