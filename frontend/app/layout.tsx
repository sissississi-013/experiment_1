import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Validation Pipeline",
  description: "Autonomous image dataset validation console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
