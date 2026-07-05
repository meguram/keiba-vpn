/** @type {import('next').NextConfig} */
const flaskApi = process.env.KEIBA_API_URL || "http://127.0.0.1:5000";

const nextConfig = {
  env: {
    KEIBA_API_URL: flaskApi,
  },
  async rewrites() {
    return [
      { source: "/api/v1/:path*", destination: `${flaskApi}/api/v1/:path*` },
    ];
  },
};

module.exports = nextConfig;
