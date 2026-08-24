/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: '#07090C',
          900: '#0B0D11',
          850: '#101318',
          800: '#161A21',
          750: '#1C222B',
          700: '#242C37',
          600: '#323D4D',
          500: '#475569',
        },
        surface: {
          base: '#0B0D11',
          panel: '#101318',
          elevated: '#161A21',
          highlight: '#1C222B',
          border: '#242C37',
          borderStrong: '#323D4D',
        },
        state: {
          observe: '#94A3B8',
          review: '#F59E0B',
          escalated: '#EF4444',
          normal: '#10B981',
          live: '#22C55E',
          tech: '#38BDF8',
        },
        telemetry: {
          muted: '#8E9BAE',
          dim: '#546274',
          bright: '#E6ECF5',
        }
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['"IBM Plex Mono"', '"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      fontSize: {
        '2xs': '0.65rem',
      },
      boxShadow: {
        'soc-panel': '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px -1px rgba(0, 0, 0, 0.4)',
        'soc-elevated': '0 4px 12px 0 rgba(0, 0, 0, 0.5)',
        'hud-badge': 'inset 0 0 0 1px rgba(255, 255, 255, 0.08)',
      }
    },
  },
  plugins: [],
}
