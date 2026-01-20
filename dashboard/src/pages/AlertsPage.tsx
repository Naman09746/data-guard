import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    Bell,
    AlertTriangle,
    AlertCircle,
    Info,
    CheckCircle,
    Clock,
    Check,
    X,
    Loader2,
    Filter,
} from 'lucide-react'
import { getAlerts, acknowledgeAlert, resolveAlert, type Alert, type AlertsResponse } from '../api/client'

export default function AlertsPage() {
    const [statusFilter, setStatusFilter] = useState<string>('open')
    const queryClient = useQueryClient()

    const { data, isLoading, error } = useQuery<AlertsResponse>({
        queryKey: ['alerts', statusFilter],
        queryFn: () => getAlerts(statusFilter || undefined),
        refetchInterval: 30000, // Refresh every 30 seconds
    })

    const acknowledgeMutation = useMutation({
        mutationFn: acknowledgeAlert,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['alerts'] }),
    })

    const resolveMutation = useMutation({
        mutationFn: (alertId: string) => resolveAlert(alertId),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['alerts'] }),
    })

    const getSeverityIcon = (severity: string) => {
        switch (severity) {
            case 'critical': return <AlertTriangle className="w-5 h-5 text-red-400" />
            case 'error': return <AlertCircle className="w-5 h-5 text-orange-400" />
            case 'warning': return <AlertTriangle className="w-5 h-5 text-amber-400" />
            default: return <Info className="w-5 h-5 text-blue-400" />
        }
    }

    const getSeverityBg = (severity: string) => {
        switch (severity) {
            case 'critical': return 'bg-red-500/20 border-red-500/30'
            case 'error': return 'bg-orange-500/20 border-orange-500/30'
            case 'warning': return 'bg-amber-500/20 border-amber-500/30'
            default: return 'bg-blue-500/20 border-blue-500/30'
        }
    }

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'open': return <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">Open</span>
            case 'acknowledged': return <span className="px-2 py-1 rounded-full text-xs bg-amber-500/20 text-amber-400">Acknowledged</span>
            case 'resolved': return <span className="px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400">Resolved</span>
            default: return <span className="px-2 py-1 rounded-full text-xs bg-slate-500/20 text-slate-400">{status}</span>
        }
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Bell className="w-8 h-8 text-amber-400" />
                        Alerts
                    </h1>
                    <p className="text-slate-400">
                        Monitor and manage data quality and leakage alerts
                    </p>
                </div>
            </div>

            {/* Summary Cards */}
            {data?.summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="glass-card p-4">
                        <p className="text-sm text-slate-400">Total Alerts</p>
                        <p className="text-2xl font-bold text-white">{data.summary.total_alerts}</p>
                    </div>
                    <div className="glass-card p-4 border border-red-500/30 bg-red-500/5">
                        <p className="text-sm text-slate-400">Open</p>
                        <p className="text-2xl font-bold text-red-400">{data.summary.open_alerts}</p>
                    </div>
                    <div className="glass-card p-4 border border-amber-500/30 bg-amber-500/5">
                        <p className="text-sm text-slate-400">Critical Open</p>
                        <p className="text-2xl font-bold text-amber-400">{data.summary.critical_open}</p>
                    </div>
                    <div className="glass-card p-4">
                        <p className="text-sm text-slate-400">By Type</p>
                        <div className="flex gap-2 mt-1">
                            {Object.entries(data.summary.by_type || {}).map(([type, count]) => (
                                <span key={type} className="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300">
                                    {type}: {count}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Filter */}
            <div className="flex items-center gap-4">
                <Filter className="w-5 h-5 text-slate-400" />
                <div className="flex gap-2">
                    {['all', 'open', 'acknowledged', 'resolved'].map((status) => (
                        <button
                            key={status}
                            onClick={() => setStatusFilter(status === 'all' ? '' : status)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${(status === 'all' && !statusFilter) || statusFilter === status
                                    ? 'bg-indigo-500 text-white'
                                    : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
                                }`}
                        >
                            {status.charAt(0).toUpperCase() + status.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            {/* Loading */}
            {isLoading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="glass-card p-6 border border-red-500/30 bg-red-500/10">
                    <p className="text-red-400">Failed to load alerts. Make sure the API server is running.</p>
                </div>
            )}

            {/* Alerts List */}
            {data?.alerts && (
                <div className="space-y-4">
                    {data.alerts.length === 0 ? (
                        <div className="glass-card p-12 text-center">
                            <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
                            <p className="text-white font-medium">No alerts found</p>
                            <p className="text-slate-400 text-sm">All clear! No matching alerts.</p>
                        </div>
                    ) : (
                        data.alerts.map((alert: Alert) => (
                            <div
                                key={alert.alert_id}
                                className={`glass-card p-6 border ${getSeverityBg(alert.severity)}`}
                            >
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex items-start gap-4">
                                        {getSeverityIcon(alert.severity)}
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3 mb-2">
                                                <h3 className="text-white font-medium">{alert.title}</h3>
                                                {getStatusBadge(alert.status)}
                                            </div>
                                            <p className="text-slate-300 text-sm">{alert.message}</p>
                                            <div className="flex items-center gap-4 mt-3 text-xs text-slate-400">
                                                <span className="flex items-center gap-1">
                                                    <Clock className="w-3 h-3" />
                                                    {new Date(alert.created_at).toLocaleString()}
                                                </span>
                                                <span>Source: {alert.source}</span>
                                                {alert.affected_features.length > 0 && (
                                                    <span>Features: {alert.affected_features.join(', ')}</span>
                                                )}
                                            </div>

                                            {/* Recommendations */}
                                            {alert.recommendations.length > 0 && (
                                                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                                                    <p className="text-xs text-slate-400 mb-2">Recommended Actions:</p>
                                                    <ul className="space-y-1">
                                                        {alert.recommendations.slice(0, 2).map((rec, idx) => (
                                                            <li key={idx} className="text-sm text-indigo-300">
                                                                • {rec.description}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Actions */}
                                    {alert.status === 'open' && (
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => acknowledgeMutation.mutate(alert.alert_id)}
                                                disabled={acknowledgeMutation.isPending}
                                                className="p-2 rounded-lg bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors"
                                                title="Acknowledge"
                                            >
                                                <Check className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => resolveMutation.mutate(alert.alert_id)}
                                                disabled={resolveMutation.isPending}
                                                className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors"
                                                title="Resolve"
                                            >
                                                <X className="w-4 h-4" />
                                            </button>
                                        </div>
                                    )}
                                    {alert.status === 'acknowledged' && (
                                        <button
                                            onClick={() => resolveMutation.mutate(alert.alert_id)}
                                            disabled={resolveMutation.isPending}
                                            className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors"
                                            title="Resolve"
                                        >
                                            <Check className="w-4 h-4" />
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    )
}
