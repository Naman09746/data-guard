import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
    ShieldCheck,
    AlertTriangle,
    Clock,
    Database,
    TrendingUp,
    FileCheck,
    ArrowRight,
    Activity,
    Zap,
    ShieldAlert,
    BarChart3,
    ArrowUpRight,
    Loader2
} from 'lucide-react'
import { 
    AreaChart, 
    Area, 
    XAxis, 
    YAxis, 
    Tooltip, 
    ResponsiveContainer,
    BarChart,
    Bar,
    Cell
} from 'recharts'
import { motion } from 'framer-motion'
import { getDashboardStats, type DashboardStats } from '../api/client'

export default function DashboardPage() {
    const [stats, setStats] = useState<DashboardStats | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function fetchStats() {
            try {
                const data = await getDashboardStats()
                setStats(data)
            } catch (error) {
                console.error("Failed to fetch dashboard stats", error)
            } finally {
                setLoading(false)
            }
        }
        fetchStats()
    }, [])

    if (loading) {
        return (
            <div className="h-[80vh] flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
            </div>
        )
    }

    const hasData = stats && stats.summary.total_scans > 0

    const statCards = stats ? [
        {
            title: 'Global Health Score',
            value: `${stats.summary.global_health_score}%`,
            desc: 'Aggregate system integrity',
            icon: Activity,
            color: 'text-indigo-400',
            bg: 'bg-indigo-500/10'
        },
        {
            title: 'Risks Blocked',
            value: stats.summary.total_issues_blocked.toLocaleString(),
            desc: 'Total issues identified',
            icon: ShieldAlert,
            color: 'text-rose-400',
            bg: 'bg-rose-500/10'
        },
        {
            title: 'Active Alerts',
            value: stats.summary.active_alerts.toString(),
            desc: 'Requires attention',
            icon: AlertTriangle,
            color: 'text-amber-400',
            bg: 'bg-amber-500/10'
        },
        {
            title: 'Datasets Scanned',
            value: stats.summary.total_scans.toString(),
            desc: 'Production coverage',
            icon: Database,
            color: 'text-emerald-400',
            bg: 'bg-emerald-500/10'
        }
    ] : []

    const riskData = stats ? Object.entries(stats.risk_distribution).map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value
    })) : []

    const COLORS = ['#10b981', '#f59e0b', '#f43f5e', '#6366f1']

    return (
        <div className="space-y-8 pb-12">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-black text-white mb-2 flex items-center gap-3">
                        <Activity className="w-8 h-8 text-indigo-500" />
                        Executive Command Center
                    </h1>
                    <p className="text-slate-400">Enterprise-wide ML Observability & Data Integrity</p>
                </div>
                <div className="flex gap-4">
                    <button className="px-4 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 font-bold hover:text-white transition-all">
                        Export Global Audit (PDF)
                    </button>
                </div>
            </div>

            {!hasData ? (
                <div className="glass-card p-20 text-center max-w-2xl mx-auto">
                    <div className="w-24 h-24 rounded-3xl bg-indigo-500/10 flex items-center justify-center mx-auto mb-8">
                        <ShieldCheck className="w-12 h-12 text-indigo-400" />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-4">No Intelligence Data</h2>
                    <p className="text-slate-400 mb-8">Your command center will populate once you start scanning datasets for quality, drift, or leakage.</p>
                    <div className="flex justify-center gap-4">
                        <Link to="/quality" className="px-8 py-4 rounded-2xl bg-indigo-500 text-white font-bold hover:scale-[1.02] transition-all">
                            Start Quality Scan
                        </Link>
                    </div>
                </div>
            ) : (
                <>
                    {/* Stat Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {statCards.map((stat, i) => (
                            <motion.div 
                                key={stat.title}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className="glass-card p-6"
                            >
                                <div className="flex items-center justify-between mb-4">
                                    <div className={`p-3 rounded-2xl ${stat.bg}`}>
                                        <stat.icon className={`w-6 h-6 ${stat.color}`} />
                                    </div>
                                    <div className="flex items-center gap-1 text-emerald-400 text-xs font-bold">
                                        <ArrowUpRight className="w-3 h-3" />
                                        12%
                                    </div>
                                </div>
                                <h3 className="text-slate-500 text-sm font-bold uppercase tracking-tighter mb-1">{stat.title}</h3>
                                <p className="text-3xl font-black text-white mb-1">{stat.value}</p>
                                <p className="text-slate-400 text-xs">{stat.desc}</p>
                            </motion.div>
                        ))}
                    </div>

                    {/* Main Charts Row */}
                    <div className="grid grid-cols-12 gap-8">
                        {/* Health Trend */}
                        <div className="col-span-12 lg:col-span-8 glass-card p-8">
                            <div className="flex items-center justify-between mb-8">
                                <div>
                                    <h3 className="text-xl font-bold text-white mb-1">Health Integrity Trend</h3>
                                    <p className="text-slate-400 text-sm">System-wide data quality stability over time</p>
                                </div>
                                <TrendingUp className="w-6 h-6 text-indigo-400" />
                            </div>
                            <div className="h-80 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={stats.timeline}>
                                        <defs>
                                            <linearGradient id="healthGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                                                <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <XAxis dataKey="date" stroke="#475569" fontSize={10} tickFormatter={(val) => val.split('-').slice(1).join('/')} />
                                        <YAxis stroke="#475569" fontSize={10} domain={[0, 100]} />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                            itemStyle={{ color: '#fff', fontWeight: 'bold' }}
                                        />
                                        <Area type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#healthGradient)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Risk Distribution */}
                        <div className="col-span-12 lg:col-span-4 glass-card p-8">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="text-xl font-bold text-white">Risk Exposure</h3>
                                <BarChart3 className="w-6 h-6 text-rose-400" />
                            </div>
                            <div className="h-80 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={riskData} layout="vertical">
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={12} width={80} />
                                        <Tooltip cursor={{fill: 'transparent'}} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                        <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={32}>
                                            {riskData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Recent Critical Scans */}
                    <div className="glass-card p-8">
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="text-xl font-bold text-white">Recent Data Operations</h3>
                            <Link to="/history" className="text-indigo-400 text-sm font-bold flex items-center gap-1 hover:underline">
                                View Full Audit Trail
                                <ArrowRight className="w-4 h-4" />
                            </Link>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="text-left text-xs font-bold text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-4">
                                        <th className="pb-4">Dataset Artifact</th>
                                        <th className="pb-4">Operation</th>
                                        <th className="pb-4">Integrity Score</th>
                                        <th className="pb-4">Timestamp</th>
                                        <th className="pb-4 text-right">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.recent_scans.map((scan) => (
                                        <tr key={scan.id} className="border-b border-slate-800/50 group">
                                            <td className="py-5">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 rounded-lg bg-slate-800 text-slate-400 group-hover:text-white transition-colors">
                                                        <FileCheck className="w-5 h-5" />
                                                    </div>
                                                    <span className="text-white font-bold">{scan.name}</span>
                                                </div>
                                            </td>
                                            <td className="py-5">
                                                <span className="px-3 py-1 rounded-lg bg-slate-800 text-slate-400 text-xs font-bold uppercase">
                                                    {scan.type}
                                                </span>
                                            </td>
                                            <td className="py-5">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                        <div 
                                                            className={`h-full ${scan.score && scan.score > 80 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                                            style={{ width: `${scan.score || 0}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm font-bold text-white">{scan.score ? `${scan.score}%` : 'N/A'}</span>
                                                </div>
                                            </td>
                                            <td className="py-5 text-slate-500 text-sm">
                                                {new Date(scan.date).toLocaleDateString()}
                                            </td>
                                            <td className="py-5 text-right">
                                                <Link to={scan.type === 'eda' ? '/eda' : scan.type === 'drift' ? '/drift' : '/leakage'} className="p-2 rounded-lg hover:bg-indigo-500/10 text-slate-500 hover:text-indigo-400 transition-all inline-block">
                                                    <Maximize2 className="w-5 h-5" />
                                                </Link>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    )
}

function Maximize2(props: any) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
    )
}
