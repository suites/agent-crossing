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
  createdAt: string;
  lastAccessedAt: string;
  importanceScore: number;
  type: 'observation' | 'interaction' | 'reflection';
  metadata?: Record<string, any>;
}

export interface SpatialNode {
  id: string;
  name: string;
  type: 'town' | 'building' | 'room' | 'object';
  parentId?: string;
  position?: { x: number; y: number };
}
