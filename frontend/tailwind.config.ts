import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        display: ["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "Inter", "sans-serif"],
        body: ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "sans-serif"],
        mono: ["SF Mono", "Menlo", "monospace"],
      },
      colors: {
        apple: {
          bg: { light: "#FFFFFF", dark: "#000000" },
          surface: { light: "#F5F5F7", dark: "#1C1C1E" },
          elevated: { light: "#FFFFFF", dark: "#2C2C2E" },
          accent: { DEFAULT: "#0071E3", hover: "#0077ED" },
          success: "#34C759",
          warning: "#FF9F0A",
          error: "#FF3B30",
        },
      },
      borderRadius: { apple: "12px", pill: "980px" },
      maxWidth: { content: "980px", wide: "1440px" },
      transitionTimingFunction: { apple: "cubic-bezier(0.25, 0.1, 0.25, 1)" },
    },
  },
  plugins: [],
};
export default config;
