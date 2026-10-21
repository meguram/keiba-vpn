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

// dev/stg/prod でビルドキャッシュディレクトリを分離し、共有による破損を防ぐ。
// NEXT_DIST_DIR で上書きできる（稼働中のサーバを止めずに別ディレクトリへビルドし、成功後に入れ替えるため）。
const distDir =
  process.env.NEXT_DIST_DIR || (env === "stg" ? ".next-stg" : env === "prod" ? ".next-prod" : ".next-dev");

// ビルドの並列数。既定の「コア数ぶんのワーカー」だと、24 コアの共有ホストで約 25 プロセスが走り負荷が急上昇した。
// stg は 2 に制限（NEXT_BUILD_CPUS で変更可。dev / prod は Next の既定のまま）。
const buildCpus = Number(process.env.NEXT_BUILD_CPUS) || (env === "stg" ? 2 : undefined);

const nextConfig = {
  distDir,
  ...(buildCpus
    ? { experimental: { cpus: buildCpus, staticGenerationMaxConcurrency: buildCpus, workerThreads: false } }
    : {}),
  env: {
    KEIBA_API_URL: flaskApi,
    KEIBA_FASTAPI_URL: fastApi,
    NEXT_PUBLIC_MONITOR_PORT: portOf("KEIBA_PORT_MONITOR"),
  },
  // 検索エンジン・AI クローラーに索引・複製させない（robots.txt は守る相手にしか効かないため、ヘッダでも明示）
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Robots-Tag", value: "noindex, nofollow, noarchive, noai, noimageai" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Content-Type-Options", value: "nosniff" },
        ],
      },
    ];
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
