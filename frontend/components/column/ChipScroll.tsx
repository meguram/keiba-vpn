"use client";

import { useEffect } from "react";

/** スマホのカテゴリのチップで、選んでいるチップを横に流して見える位置へ送る（縦には動かさない） */
export function ChipScroll() {
  useEffect(() => {
    document.querySelectorAll<HTMLElement>(".col-chips").forEach((row) => {
      const on = row.querySelector<HTMLElement>("[aria-current]");
      if (on) row.scrollLeft = on.offsetLeft - row.offsetLeft - 16;
    });
  }, []);
  return null;
}
