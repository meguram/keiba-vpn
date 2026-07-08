"use client";
import { useState, useEffect } from "react";

export type AuthStatus = {
  loading: boolean;
  loggedIn: boolean;
  isAdmin: boolean;
  isMember: boolean;
};

export function useAuthStatus(): AuthStatus {
  const [status, setStatus] = useState<AuthStatus>({
    loading: true,
    loggedIn: false,
    isAdmin: false,
    isMember: false,
  });

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_MOCK === "true") {
      setStatus({ loading: false, loggedIn: false, isAdmin: false, isMember: false });
      return;
    }
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) =>
        r.ok ? r.json() : { logged_in: false, is_admin: false, is_member: false }
      )
      .then((d) =>
        setStatus({
          loading: false,
          loggedIn: !!d.logged_in,
          isAdmin: !!d.is_admin,
          isMember: !!d.is_member,
        })
      )
      .catch(() =>
        setStatus({ loading: false, loggedIn: false, isAdmin: false, isMember: false })
      );
  }, []);

  return status;
}
