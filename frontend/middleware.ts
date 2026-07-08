import { NextRequest, NextResponse } from "next/server";

// Pages accessible without membership
const PUBLIC_PATHS = [
  "/",
  "/races",
  "/login",
  "/api/",     // all API routes pass through
  "/_next/",   // Next.js internals
  "/favicon",
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isPublic = PUBLIC_PATHS.some(
    (p) => pathname === p || pathname.startsWith(p)
  );
  if (isPublic) return NextResponse.next();

  // For member-only pages: the actual auth check happens client-side.
  // Middleware only handles the cookie presence check (fast edge check).
  const sessionCookie = request.cookies.get("keiba_dev_session");
  if (!sessionCookie) {
    // Redirect to home with modal flag
    const url = request.nextUrl.clone();
    url.pathname = "/";
    url.searchParams.set("upgrade", "1");
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
