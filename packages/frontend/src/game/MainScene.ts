import Phaser from "phaser";

export class MainScene extends Phaser.Scene {
  constructor() {
    super("MainScene");
  }

  preload() {
    // Placeholder for assets
  }

  create() {
    this.add
      .text(400, 300, "Agent Crossing: Neural Horizons", {
        fontSize: "32px",
        color: "#ffffff",
      })
      .setOrigin(0.5);

    this.add
      .text(400, 350, "Phaser 3 + React 19 Initialized", {
        fontSize: "18px",
        color: "#aaaaaa",
      })
      .setOrigin(0.5);
  }

  update() {
    // Game loop logic
  }
}
