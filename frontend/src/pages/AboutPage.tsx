import { ExternalLink, GitBranch, Headphones, Palette, ScanEye } from "lucide-react";

const highlights = [
  ["NASA imagery", ScanEye],
  ["Computer vision", Palette],
  ["Live MIDI sound", Headphones],
] as const;

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-7xl rounded-2xl bg-[#E5CD82]/60 p-6 shadow-md sm:p-8">
      <p className="eyebrow">About TerraGroove</p>
      <h1 className="mt-2 font-display text-4xl text-white text-shadow-md">Listen to the planet.</h1>
      <p className="mt-4 text-lg leading-relaxed text-black/40">
        TerraGroove turns NASA Earth observation imagery into a tactile, musical
        landscape. Explore a dataset with the trackpad and the live Python
        sequencer responds to color, texture, intensity, and movement.
      </p>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <section className="rounded-2xl bg-[#FFF2B4] p-4 shadow-md">
          <p className="eyebrow">NASA Space Apps Challenge</p>
          <h2 className="mt-2 font-display text-2xl text-white text-shadow-md">
            Earth Information Jukebox
          </h2>
          <a
            className="mt-4 inline-flex items-center gap-2 rounded-full bg-[#15370D] px-3 py-1.5 font-semibold text-white transition hover:bg-[#728E6D]"
            href="https://github.com/GitHappens-net/EarthJukebox"
            target="_blank"
            rel="noreferrer"
          >
            <GitBranch size={18} aria-hidden="true" />
            View the project
            <ExternalLink size={15} aria-hidden="true" />
          </a>
        </section>
        <section className="rounded-2xl bg-[#FFF2B4] p-4 shadow-md">
          <p className="eyebrow">Team GitHappens</p>
          <ul className="mt-2 space-y-1 text-base text-black/40">
            {["Savvas Tzanetis", "Vasilis Zoidis", "Lydia Strantzia", "Giorgos Sivridis"].map((member) => (
              <li key={member}>{member}</li>
            ))}
          </ul>
        </section>
      </div>
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {highlights.map(([item, Icon]) => (
          <div className="flex items-center gap-3 rounded-full bg-[#FFF2B4] p-4 shadow-md" key={item}>
            <Icon size={20} aria-hidden="true" />
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}
