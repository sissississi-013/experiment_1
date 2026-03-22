"use client";
import { useRouter } from "next/navigation";

export default function LiveIndexPage() {
  const router = useRouter();
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "80vh", flexDirection: "column", gap: "16px" }}>
      <p style={{ color: "#6E6E73", fontSize: "15px" }}>No active run. Start a new validation to see live progress.</p>
      <button
        onClick={() => router.push("/new-run")}
        style={{
          background: "#0071E3", color: "white", border: "none",
          borderRadius: "980px", padding: "10px 24px", fontSize: "15px",
          fontWeight: 500, cursor: "pointer", fontFamily: "inherit",
        }}
      >
        New Run
      </button>
    </div>
  );
}
