import { useQuery } from '@tanstack/react-query'
import {
    History,
    Database,
    Clock,
    CheckCircle,
    AlertCircle,
    TrendingUp,
    TrendingDown,
    Loader2,
    FileText,
} from 'lucide-react'
import { getScanHistory, type ScanHistoryResponse, type ScanRecord } from '../api/client'

export default function HistoryPage() {
    const { data, isLoading, error } = useQuery<ScanHistoryResponse>({
        queryKey: ['scanHistory'],
        queryFn: () => getScanHistory(50),
        refetchInterval: 60000, // Refresh every minute
    })

    const formatDate = (dateStr: string) => {
        return new Date(dateStr).toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    const getStatusBadge = (status: string) => {
        if (status === 'passed' || status === 'clean') {
            return (
                <span className="flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400">
                    <CheckCircle className="w-3 h-3" />
                    Passed
                </span>
            )
        }
        return (
            <span className="flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">
                <AlertCircle className="w-3 h-3" />
                Issues
            </span>
        )
    }

    const getScanTypeBadge = (type: string) => {
        const colors: Record<string, string> = {
            quality: 'bg-indigo-500/20 text-indigo-400',
            leakage: 'bg-purple-500/20 text-purple-400',
            full: 'bg-amber-500/20 text-amber-400',
        }
        return (
            <span className={`px-2 py-1 rounded-full text-xs ${colors[type] || 'bg-slate-500/20 text-slate-400'}`}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
            </span>
        )
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                    <History className="w-8 h-8 text-cyan-400" />
                    Scan History
                </h1>
                <p className="text-slate-400">
                    Track dataset versions and scan results over time
                </p>
            </div>

            {/* Stats Cards */}
            {data && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <FileText className="w-4 h-4 text-slate-400" />
                            <span className="text-sm text-slate-400">Total Scans</span>
                        </div>
                        <p className="text-2xl font-bold text-white">{data.total}</p>
                    </div>
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                            <span className="text-sm text-slate-400">Passed</span>
                        </div>
                        <p className="text-2xl font-bold text-emerald-400">
                            {data.scans.filter(s => s.status === 'passed' || s.status === 'clean').length}
                        </p>
                    </div>
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <AlertCircle className="w-4 h-4 text-red-400" />
                            <span className="text-sm text-slate-400">With Issues</span>
                        </div>
                        <p className="text-2xl font-bold text-red-400">
                            {data.scans.filter(s => s.status !== 'passed' && s.status !== 'clean').length}
                        </p>
                    </div>
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Database className="w-4 h-4 text-cyan-400" />
                            <span className="text-sm text-slate-400">Unique Datasets</span>
                        </div>
                        <p className="text-2xl font-bold text-cyan-400">
                            {new Set(data.scans.map(s => s.dataset_version.version_hash)).size}
                        </p>
                    </div>
                </div>
            )}

            {/* Loading */}
            {isLoading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="glass-card p-6 border border-red-500/30 bg-red-500/10">
                    <p className="text-red-400">Failed to load scan history. Make sure the API server is running.</p>
                </div>
            )}

            {/* History Table */}
            {data?.scans && (
                <div className="glass-card overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-700/50">
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Timestamp</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Type</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Dataset</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Status</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Issues</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Quality</th>
                                    <th className="px-6 py-4 text-left text-sm font-medium text-slate-400">Duration</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-700/50">
                                {data.scans.length === 0 ? (
                                    <tr>
                                        <td colSpan={7} className="px-6 py-12 text-center">
                                            <History className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                                            <p className="text-slate-400">No scan history yet</p>
                                            <p className="text-sm text-slate-500">Run a quality or leakage scan to start tracking</p>
                                        </td>
                                    </tr>
                                ) : (
                                    data.scans.map((scan: ScanRecord) => (
                                        <tr key={scan.scan_id} className="hover:bg-slate-700/30 transition-colors">
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-2 text-white">
                                                    <Clock className="w-4 h-4 text-slate-400" />
                                                    {formatDate(scan.timestamp)}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                {getScanTypeBadge(scan.scan_type)}
                                            </td>
                                            <td className="px-6 py-4">
                                                <div>
                                                    <p className="text-white font-mono text-sm">{scan.dataset_version.version_hash}</p>
                                                    <p className="text-xs text-slate-400">
                                                        {scan.dataset_version.row_count.toLocaleString()} rows × {scan.dataset_version.column_count} cols
                                                    </p>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                {getStatusBadge(scan.status)}
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`font-medium ${scan.total_issues > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                                    {scan.total_issues}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                {scan.quality_score !== null ? (
                                                    <div className="flex items-center gap-2">
                                                        {scan.quality_score >= 0.8 ? (
                                                            <TrendingUp className="w-4 h-4 text-emerald-400" />
                                                        ) : (
                                                            <TrendingDown className="w-4 h-4 text-amber-400" />
                                                        )}
                                                        <span className={`font-medium ${scan.quality_score >= 0.8 ? 'text-emerald-400' : 'text-amber-400'
                                                            }`}>
                                                            {(scan.quality_score * 100).toFixed(0)}%
                                                        </span>
                                                    </div>
                                                ) : (
                                                    <span className="text-slate-500">—</span>
                                                )}
                                            </td>
                                            <td className="px-6 py-4 text-slate-400 text-sm">
                                                {scan.duration_seconds.toFixed(2)}s
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}
