/** @type {import('next').NextConfig} */
const fs = require("fs");
const path = require("path");

// ポートの正は ../config/ports.env（環境変数 KEIBA_PORT_* が優先）
function loadPorts() {
  const ports = {};
  try {
    const text = fs.readFileSync(path.join(__dirname, "..", "config", "ports.env"), "utf8");
    for (const raw of text.split("\n")) {
      const line = raw.split("#")[0].trim();
      const m = line.match(/^(KEIBA_PORT_[A-Z0-9_]+)=(\d+)$/);
      if (m) ports[m[1]] = m[2];
    }
  } catch (_) {
    /* ports.env が無い場合は環境変数のみ */
  }
  return (key) => process.env[key] || ports[key];
}
const portOf = loadPorts();

const env = process.env.KEIBA_ENV || "dev";
const envKey = env === "stg" || env === "prod" ? env : "dev";
const flaskApi =
  process.env.KEIBA_API_URL || `http://127.0.0.1:${portOf(`KEIBA_PORT_${envKey.toUpperCase()}_FLASK`)}`;
const fastApi =
  process.env.KEIBA_FASTAPI_URL || `http://127.0.0.1:${portOf(`KEIBA_PORT_${envKey.toUpperCase()}_FASTAPI`)}`;

// dev/stg/prod でビルドキャッシュディレクトリを分離し、共有による破損を防ぐ
const distDir = env === "stg" ? ".next-stg" : env === "prod" ? ".next-prod" : ".next-dev";

const nextConfig = {
  distDir,
  env: {
    KEIBA_API_URL: flaskApi,
    NEXT_PUBLIC_MONITOR_PORT: portOf("KEIBA_PORT_MONITOR"),
  },
  async rewrites() {
    return [
      { source: "/api/v1/:path*", destination: `${flaskApi}/api/v1/:path*` },
      { source: "/api/:path*", destination: `${fastApi}/api/:path*` },
      { source: "/data/image/:path*", destination: `${fastApi}/data/image/:path*` },
    ];
  },
};

module.exports = nextConfig;
