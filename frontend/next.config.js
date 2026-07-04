/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const api = process.env.KEIBA_API_URL || "http://127.0.0.1:5000";
    return [
      { source: "/api/v1/:path*", destination: `${api}/api/v1/:path*` },
    ];
  },
};

module.exports = nextConfig;
