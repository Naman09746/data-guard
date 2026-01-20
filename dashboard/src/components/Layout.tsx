import type { ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
    LayoutDashboard,
    ShieldCheck,
    AlertTriangle,
    Settings,
    Database,
    Activity,
    Brain,
    FlaskConical,
    Bell,
    History,
} from 'lucide-react'

interface LayoutProps {
    children: ReactNode
}

const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/quality', label: 'Data Quality', icon: ShieldCheck },
    { path: '/leakage', label: 'Leakage Detection', icon: AlertTriangle },
    { path: '/risk-scoring', label: 'Risk Scoring', icon: Brain },
    { path: '/experiments', label: 'Experiments', icon: FlaskConical },
    { path: '/alerts', label: 'Alerts', icon: Bell },
    { path: '/history', label: 'History', icon: History },
    { path: '/rules', label: 'Custom Rules', icon: Settings },
]

export default function Layout({ children }: LayoutProps) {
    const location = useLocation()

    return (
        <div className="flex min-h-screen">
            {/* Sidebar */}
            <aside className="fixed left-0 top-0 h-screen w-64 glass-card border-r border-slate-700/50 flex flex-col">
                {/* Logo */}
                <div className="p-6 border-b border-slate-700/50">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                            <Database className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="font-bold text-white text-lg">DataGuard</h1>
                            <p className="text-xs text-slate-400">Quality & Leakage</p>
                        </div>
                    </div>
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 space-y-2">
                    {navItems.map((item) => {
                        const Icon = item.icon
                        const isActive = location.pathname === item.path

                        return (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group ${isActive
                                    ? 'bg-gradient-to-r from-indigo-500/20 to-purple-500/20 text-white border border-indigo-500/30'
                                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                                    }`}
                            >
                                <Icon
                                    className={`w-5 h-5 transition-transform group-hover:scale-110 ${isActive ? 'text-indigo-400' : ''
                                        }`}
                                />
                                <span className="font-medium">{item.label}</span>
                                {isActive && (
                                    <div className="ml-auto w-2 h-2 rounded-full bg-indigo-400 pulse-glow" />
                                )}
                            </NavLink>
                        )
                    })}
                </nav>

                {/* Status */}
                <div className="p-4 border-t border-slate-700/50">
                    <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                        <Activity className="w-5 h-5 text-emerald-400" />
                        <div>
                            <p className="text-sm font-medium text-emerald-400">API Connected</p>
                            <p className="text-xs text-slate-400">localhost:8000</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main content */}
            <main className="flex-1 ml-64 p-8">
                {children}
            </main>
        </div>
    )
}
