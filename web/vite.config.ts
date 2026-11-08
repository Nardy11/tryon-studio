import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

// The API runs on :8000 (uvicorn). In dev, Vite proxies /api so the browser sees one origin.
// https by default (a self-signed certificate): browsers only allow the camera, used by hand
// gestures, on https pages or on localhost. On a phone, accept the certificate warning once.
// TRYON_HTTP=1 serves plain http instead; the page works, and the camera then only on localhost.
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const http = env.TRYON_HTTP === '1';

export default defineConfig({
  // GitHub Pages serves the site under /<repo>/; the deploy workflow sets VITE_BASE.
  base: env.VITE_BASE || '/',
  plugins: http ? [react()] : [react(), basicSsl()],
  server: {
    port: 5175,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
});
