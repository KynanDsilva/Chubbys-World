import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "chubby.world — Autonomous Scrapyard",
  description:
    "Explore Chubby's 32x32 autonomous scrapyard in first-person 3D, leave pixel offerings, and approve inspired blueprints.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
