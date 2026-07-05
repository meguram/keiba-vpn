/** @type {import('next').NextConfig} */
const flaskApi = process.env.KEIBA_API_URL || "http://127.0.0.1:5000";
const fastApi = process.env.KEIBA_FASTAPI_URL || "http://127.0.0.1:8000";

const nextConfig = {
  env: {
    KEIBA_API_URL: flaskApi,
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
