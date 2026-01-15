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
    Trash2,
} from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts'
import { getStats, clearHistory, type ValidationRecord } from '../store/validationHistory'

export default function DashboardPage() {
    const [stats, setStats] = useState(getStats())

    useEffect(() => {
        // Refresh stats when page loads or when returning to it
        setStats(getStats())
    }, [])

    const handleClearHistory = () => {
        if (confirm('Clear all validation history?')) {
            clearHistory()
            setStats(getStats())
        }
    }

    const formatTime = (date: Date | null) => {
        if (!date) return 'Never'
        const diff = Date.now() - date.getTime()
        const mins = Math.floor(diff / 60000)
        if (mins < 1) return 'Just now'
        if (mins < 60) return `${mins}m ago`
        const hours = Math.floor(mins / 60)
        if (hours < 24) return `${hours}h ago`
        return `${Math.floor(hours / 24)}d ago`
    }

    const getStatusBadge = (status: string) => {
        const styles: Record<string, string> = {
            passed: 'status-passed',
            clean: 'status-passed',
            warning: 'status-warning',
            failed: 'status-failed',
            detected: 'status-failed',
        }
        return styles[status] || 'bg-slate-600'
    }

    const hasData = stats.totalValidations > 0
    const qualityScore = stats.avgQualityScore?.toFixed(1) || '—'
    const leakageRisk = stats.leakageCount > 0 ? 'Risks Found' : stats.leakageValidations > 0 ? 'Clean' : '—'

    const statsCards = [
        {
            title: 'Avg Quality Score',
            value: hasData ? `${qualityScore}%` : '—',
            change: hasData ? `${stats.qualityValidations} validations` : 'No data yet',
            icon: ShieldCheck,
            color: 'from-emerald-500 to-teal-500',
            bgColor: 'bg-emerald-500/10',
            iconColor: '#10b981',
        },
        {
            title: 'Leakage Status',
            value: leakageRisk,
            change: hasData ? `${stats.cleanCount} clean / ${stats.leakageCount} risks` : 'No scans yet',
            icon: AlertTriangle,
            color: 'from-amber-500 to-orange-500',
            bgColor: 'bg-amber-500/10',
            iconColor: '#f59e0b',
        },
        {
            title: 'Last Scan',
            value: formatTime(stats.lastValidation),
            change: hasData ? 'Session-based tracking' : 'Upload a file to start',
            icon: Clock,
            color: 'from-blue-500 to-cyan-500',
            bgColor: 'bg-blue-500/10',
            iconColor: '#3b82f6',
        },
        {
            title: 'Total Validations',
            value: stats.totalValidations.toString(),
            change: `${stats.passedCount} passed, ${stats.warningCount} warnings`,
            icon: Database,
            color: 'from-purple-500 to-pink-500',
            bgColor: 'bg-purple-500/10',
            iconColor: '#a855f7',
        },
    ]

    const validationBreakdown = [
        { name: 'Passed', value: stats.passedCount || 1, color: '#10b981' },
        { name: 'Warning', value: stats.warningCount || 0, color: '#f59e0b' },
        { name: 'Failed', value: stats.failedCount || 0, color: '#ef4444' },
    ].filter(d => d.value > 0)

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
                    <p className="text-slate-400">
                        {hasData
                            ? 'Overview of your validation session'
                            : 'Upload files to start tracking data quality'}
                    </p>
                </div>
                {hasData && (
                    <button
                        onClick={handleClearHistory}
                        className="px-4 py-2 rounded-lg border border-slate-600 text-slate-400 hover:text-red-400 hover:border-red-500/50 flex items-center gap-2 transition-colors"
                    >
                        <Trash2 className="w-4 h-4" />
                        Clear History
                    </button>
                )}
            </div>

            {/* Empty State */}
            {!hasData && (
                <div className="glass-card p-12 text-center">
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center mx-auto mb-6">
                        <FileCheck className="w-10 h-10 text-indigo-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">No Validations Yet</h3>
                    <p className="text-slate-400 mb-6 max-w-md mx-auto">
                        Start by uploading a CSV file to validate data quality or detect leakage.
                        Your validation history will appear here.
                    </p>
                    <div className="flex gap-4 justify-center">
                        <Link
                            to="/quality"
                            className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium flex items-center gap-2 hover:opacity-90 transition-all"
                        >
                            Validate Quality
                            <ArrowRight className="w-4 h-4" />
                        </Link>
                        <Link
                            to="/leakage"
                            className="px-6 py-3 rounded-xl border border-slate-600 text-slate-300 font-medium flex items-center gap-2 hover:bg-slate-700/50 transition-colors"
                        >
                            Detect Leakage
                            <ArrowRight className="w-4 h-4" />
                        </Link>
                    </div>
                </div>
            )}

            {/* Stats Grid */}
            {hasData && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {statsCards.map((stat) => {
                            const Icon = stat.icon
                            return (
                                <div key={stat.title} className="glass-card p-6 hover-lift cursor-pointer">
                                    <div className="flex items-start justify-between mb-4">
                                        <div className={`p-3 rounded-xl ${stat.bgColor}`}>
                                            <Icon className="w-6 h-6" style={{ color: stat.iconColor }} />
                                        </div>
                                        <span className="text-xs text-slate-400">{stat.change}</span>
                                    </div>
                                    <h3 className="text-slate-400 text-sm mb-1">{stat.title}</h3>
                                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                                </div>
                            )
                        })}
                    </div>

                    {/* Charts Row */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Quality Trend */}
                        <div className="glass-card p-6">
                            <div className="flex items-center justify-between mb-6">
                                <div>
                                    <h3 className="text-lg font-semibold text-white">Quality Trend</h3>
                                    <p className="text-sm text-slate-400">Last {stats.qualityTrend.length} validations</p>
                                </div>
                                <TrendingUp className="w-5 h-5 text-emerald-400" />
                            </div>
                            {stats.qualityTrend.length > 1 ? (
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={stats.qualityTrend}>
                                            <defs>
                                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                                            <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#1e293b',
                                                    border: '1px solid #475569',
                                                    borderRadius: '8px',
                                                }}
                                                formatter={(value) => [`${value ?? 0}%`, 'Score']}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="value"
                                                stroke="#6366f1"
                                                strokeWidth={2}
                                                fill="url(#colorValue)"
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            ) : (
                                <div className="h-64 flex items-center justify-center text-slate-400">
                                    Run more validations to see trends
                                </div>
                            )}
                        </div>

                        {/* Validation Breakdown */}
                        <div className="glass-card p-6">
                            <div className="flex items-center justify-between mb-6">
                                <div>
                                    <h3 className="text-lg font-semibold text-white">Status Breakdown</h3>
                                    <p className="text-sm text-slate-400">All validations</p>
                                </div>
                                <FileCheck className="w-5 h-5 text-indigo-400" />
                            </div>
                            <div className="flex items-center gap-8">
                                <div className="h-48 w-48">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={validationBreakdown}
                                                innerRadius={50}
                                                outerRadius={70}
                                                paddingAngle={5}
                                                dataKey="value"
                                            >
                                                {validationBreakdown.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="space-y-3">
                                    {validationBreakdown.map((item) => (
                                        <div key={item.name} className="flex items-center gap-3">
                                            <div
                                                className="w-3 h-3 rounded-full"
                                                style={{ backgroundColor: item.color }}
                                            />
                                            <span className="text-slate-300 text-sm">{item.name}</span>
                                            <span className="text-white font-semibold ml-auto">{item.value}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Recent Validations */}
                    <div className="glass-card p-6">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h3 className="text-lg font-semibold text-white">Recent Validations</h3>
                                <p className="text-sm text-slate-400">Your latest data quality checks</p>
                            </div>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="text-left text-sm text-slate-400">
                                        <th className="pb-4">Dataset</th>
                                        <th className="pb-4">Type</th>
                                        <th className="pb-4">Status</th>
                                        <th className="pb-4">Issues</th>
                                        <th className="pb-4">Time</th>
                                    </tr>
                                </thead>
                                <tbody className="text-slate-300">
                                    {stats.recentValidations.map((validation: ValidationRecord) => (
                                        <tr key={validation.id} className="border-t border-slate-700/50">
                                            <td className="py-4 font-medium text-white">{validation.fileName}</td>
                                            <td className="py-4 capitalize">{validation.type}</td>
                                            <td className="py-4">
                                                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusBadge(validation.status)}`}>
                                                    {validation.status}
                                                </span>
                                            </td>
                                            <td className="py-4">{validation.totalIssues}</td>
                                            <td className="py-4 text-slate-400">
                                                {formatTime(new Date(validation.timestamp))}
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
