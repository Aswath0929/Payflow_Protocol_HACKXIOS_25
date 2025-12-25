import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute w-[500px] h-[500px] rounded-full opacity-20 blur-[100px] bg-gradient-to-br from-violet-600 to-cyan-600 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>

      <div className="relative text-center space-y-6 p-8">
        <div className="flex justify-center mb-4">
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-4xl opacity-50">
              💎
            </div>
          </div>
        </div>

        <h1 className="text-8xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
          404
        </h1>
        <h2 className="text-2xl font-semibold text-white">Page Not Found</h2>
        <p className="text-zinc-400 max-w-md">The page you&apos;re looking for doesn&apos;t exist or has been moved.</p>

        <Link
          href="/"
          className="inline-block mt-4 px-8 py-3 rounded-xl font-medium bg-gradient-to-r from-violet-600 to-cyan-600 text-white hover:opacity-90 transition-opacity"
        >
          ← Back to Home
        </Link>
      </div>
    </div>
  );
}
