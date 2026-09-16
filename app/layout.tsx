import type { Metadata } from "next";
import "./globals.css";
import "./brain-activity.css";
import "./observation.css";
import "./terminal.css";

export const metadata: Metadata = {
  title: "SwarmLab · Connectome learning experiments",
  description: "Watch experimental full-connectome training phases and a retained reference model, with live 2D embodiment and measured 3D neural activity.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
