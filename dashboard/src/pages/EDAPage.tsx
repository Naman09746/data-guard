import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { 
    Search, 
    Upload, 
    FileText, 
    AlertCircle, 
    CheckCircle2, 
    ChevronRight, 
    BarChart2, 
    Layout, 
    Database, 
    Zap,
    Download
} from 'lucide-react'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { profileDataset, type EDAReport, type ColumnProfile } from '../api/client'
import axios from 'axios'
const api = axios.create({ baseURL: 'http://localhost:8000/api/v1' })

export default function EDAPage() {
    const [file, setFile] = useState<File | null>(null)
    const [targetColumn, setTargetColumn] = useState('')
    const [report, setReport] = useState<EDAReport | null>(null)
    const [selectedColumn, setSelectedColumn] = useState<ColumnProfile | null>(null)
    const [taskId, setTaskId] = useState<string | null>(null)
    const [progress, setProgress] = useState(0)
    const [taskStatus, setTaskStatus] = useState<string | null>(null)

    const mutation = useMutation({
        mutationFn: ({ file, target }: { file: File; target?: string }) => 
            profileDataset(file, target),
        onSuccess: (data: any) => {
            setTaskId(data.task_id)
            setTaskStatus('processing')
            setProgress(0)
            toast.success('Analysis started in background')
        },
        onError: (error: any) => {
            toast.error(error.response?.data?.detail || 'Failed to profile dataset')
        }
    })

    useEffect(() => {
        let interval: any
        if (taskId && taskStatus === 'processing') {
            interval = setInterval(async () => {
                try {
                    const response = await api.get(`/tasks/${taskId}`)
                    const { status, progress, result, detail } = response.data
                    
                    setProgress(progress)
                    if (detail) setTaskStatus(detail)

                    if (status === 'SUCCESS') {
                        clearInterval(interval)
                        setTaskStatus('completed')
                        // Fetch the final report and insights
                        const [reportRes, insightRes] = await Promise.all([
                            api.get(`/eda/report/${result.scan_id}`),
                            api.get(`/eda/insights/${result.scan_id}`)
                        ])
                        
                        const fullReport = {
                            ...reportRes.data,
                            insight: insightRes.data
                        }
                        
                        setReport(fullReport)
                        setSelectedColumn(fullReport.column_profiles[0])
                        toast.success('Analysis complete!')
                    } else if (status === 'FAILURE') {
                        clearInterval(interval)
                        setTaskStatus('failed')
                        toast.error('Background analysis failed')
                    }
                } catch (e) {
                    console.error('Polling error', e)
                }
            }, 2000)
        }
        return () => clearInterval(interval)
    }, [taskId, taskStatus])

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0])
        }
    }

    const handleRunEDA = () => {
        if (!file) return
        mutation.mutate({ file, target: targetColumn })
    }

    const getCorrelationOption = () => {
        if (!report) return {}
        
        return {
            tooltip: { position: 'top' },
            grid: { height: '70%', top: '10%' },
            xAxis: {
                type: 'category',
                data: report.column_labels,
                splitArea: { show: true }
            },
            yAxis: {
                type: 'category',
                data: report.column_labels,
                splitArea: { show: true }
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%',
                inRange: {
                    color: ['#ef4444', '#f8fafc', '#22d3ee']
                }
            },
            series: [{
                name: 'Correlation',
                type: 'heatmap',
                data: report.correlation_matrix.flatMap((row, i) => 
                    row.map((val, j) => [j, i, parseFloat(val.toFixed(2))])
                ),
                label: { show: false },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }
    }

    const getColumnDistOption = (column: ColumnProfile) => {
        if (!column.histogram) return {}
        
        return {
            tooltip: { trigger: 'axis' },
            grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
            xAxis: {
                type: 'category',
                data: column.histogram.bin_edges.map(e => typeof e === 'number' ? e.toFixed(2) : e),
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
            series: [{
                data: column.histogram.counts,
                type: 'bar',
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#6366f1' },
                        { offset: 1, color: '#a855f7' }
                    ])
                }
            }]
        }
    }

    return (
        <div className="space-y-8 pb-12">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Search className="w-8 h-8 text-cyan-400" />
                        Advanced EDA Engine
                    </h1>
                    <p className="text-slate-400">Deep dataset profiling, statistical analysis, and visual intelligence</p>
                </div>
                {report && (
                    <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-500 transition-all">
                        <Download className="w-4 h-4" />
                        Export PDF Report
                    </button>
                )}
            </div>

            {!report ? (
                /* Upload Section */
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-12 max-w-2xl mx-auto text-center"
                >
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-6">
                        <Upload className="w-10 h-10 text-cyan-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">Profile New Dataset</h3>
                    <p className="text-slate-400 mb-8">Upload a CSV file to generate a comprehensive statistical profile</p>
                    
                    <div className="space-y-6">
                        <div className="relative">
                            <input 
                                type="file" 
                                accept=".csv"
                                onChange={handleFileChange}
                                className="hidden" 
                                id="file-upload"
                            />
                            <label 
                                htmlFor="file-upload"
                                className="block w-full p-8 border-2 border-dashed border-slate-700 rounded-2xl cursor-pointer hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-all"
                            >
                                <div className="flex flex-col items-center gap-2">
                                    <FileText className={`w-8 h-8 ${file ? 'text-cyan-400' : 'text-slate-500'}`} />
                                    <span className="text-slate-300 font-medium">
                                        {file ? file.name : 'Choose CSV file or drag here'}
                                    </span>
                                </div>
                            </label>
                        </div>

                        <div className="text-left">
                            <label className="block text-sm font-medium text-slate-400 mb-2">Target Column (Optional)</label>
                            <input 
                                type="text"
                                value={targetColumn}
                                onChange={(e) => setTargetColumn(e.target.value)}
                                placeholder="e.g. churn, price, label"
                                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
                            />
                        </div>

                        {taskId && taskStatus === 'processing' ? (
                            <div className="w-full space-y-3 pt-4">
                                <div className="flex justify-between text-xs font-bold text-cyan-400 uppercase tracking-widest px-1">
                                    <span>{taskStatus}</span>
                                    <span>{progress}%</span>
                                </div>
                                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <motion.div 
                                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-600"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${progress}%` }}
                                    />
                                </div>
                                <p className="text-slate-500 text-xs italic">This may take a few moments for large datasets...</p>
                            </div>
                        ) : (
                            <button 
                                onClick={handleRunEDA}
                                disabled={!file || mutation.isPending}
                                className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
                                    !file || mutation.isPending 
                                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:scale-[1.02]'
                                }`}
                            >
                                {mutation.isPending ? (
                                    <>
                                        <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                                        Initiating...
                                    </>
                                ) : (
                                    <>
                                        <Zap className="w-5 h-5" />
                                        Generate EDA Profile
                                    </>
                                )}
                            </button>
                        )}
                    </div>
                </motion.div>
            ) : (
                /* Report View */
                <div className="grid grid-cols-12 gap-8">
                    {/* Sidebar: Column List */}
                    <div className="col-span-12 lg:col-span-3 space-y-6">
                        <div className="glass-card p-4 h-[calc(100vh-200px)] flex flex-col">
                            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 px-2">Dataset Columns</h3>
                            <div className="flex-1 overflow-y-auto space-y-1 pr-2 custom-scrollbar">
                                {report.column_profiles.map((col) => (
                                    <button
                                        key={col.name}
                                        onClick={() => setSelectedColumn(col)}
                                        className={`w-full flex items-center justify-between p-3 rounded-xl transition-all ${
                                            selectedColumn?.name === col.name
                                            ? 'bg-cyan-500/10 border border-cyan-500/30 text-white'
                                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                                        }`}
                                    >
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                                                col.missing_pct > 0.1 ? 'bg-amber-500' : 'bg-emerald-500'
                                            }`} />
                                            <span className="truncate font-medium">{col.name}</span>
                                        </div>
                                        <span className="text-[10px] uppercase font-bold opacity-50">{col.type}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Main Content */}
                    <div className="col-span-12 lg:col-span-9 space-y-8">
                        {/* Summary Stats */}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            {[
                                { label: 'Overall Health', value: `${report.overall_health_score.toFixed(1)}%`, icon: CheckCircle2, color: 'text-emerald-400' },
                                { label: 'Total Rows', value: report.shape[0].toLocaleString(), icon: Layout, color: 'text-blue-400' },
                                { label: 'Duplicates', value: `${report.duplicate_rows} (${(report.duplicate_pct * 100).toFixed(1)}%)`, icon: AlertCircle, color: report.duplicate_rows > 0 ? 'text-amber-400' : 'text-slate-400' },
                                { label: 'Memory', value: `${report.memory_mb.toFixed(1)} MB`, icon: Database, color: 'text-purple-400' },
                            ].map((stat) => (
                                <div key={stat.label} className="glass-card p-6">
                                    <div className="flex items-center justify-between mb-2">
                                        <stat.icon className={`w-5 h-5 ${stat.color}`} />
                                        <span className="text-xs text-slate-500 uppercase font-bold">Snapshot</span>
                                    </div>
                                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                                    <h4 className="text-sm text-slate-400">{stat.label}</h4>
                                </div>
                            ))}
                        </div>

                        {/* Selected Column Deep Dive */}
                        <AnimatePresence mode="wait">
                            {selectedColumn && (
                                <motion.div 
                                    key={selectedColumn.name}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    className="glass-card p-8"
                                >
                                    <div className="flex items-center justify-between mb-8">
                                        <div className="flex items-center gap-4">
                                            <div className="p-3 rounded-2xl bg-cyan-500/10 border border-cyan-500/20">
                                                <BarChart2 className="w-6 h-6 text-cyan-400" />
                                            </div>
                                            <div>
                                                <h2 className="text-2xl font-bold text-white">{selectedColumn.name}</h2>
                                                <p className="text-slate-400 flex items-center gap-2">
                                                    <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] font-bold uppercase">{selectedColumn.type}</span>
                                                    • {selectedColumn.count.toLocaleString()} values
                                                    • {selectedColumn.unique_count.toLocaleString()} unique
                                                </p>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm text-slate-500 mb-1">Missing Values</div>
                                            <div className="text-lg font-bold text-white">
                                                {selectedColumn.missing_count.toLocaleString()} ({ (selectedColumn.missing_pct * 100).toFixed(1) }%)
                                            </div>
                                            <div className="w-32 h-1.5 bg-slate-800 rounded-full mt-2 overflow-hidden">
                                                <div 
                                                    className="h-full bg-cyan-500" 
                                                    style={{ width: `${selectedColumn.missing_pct * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                        {/* Histogram */}
                                        <div>
                                            <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Distribution</h4>
                                            <div className="h-64">
                                                <ReactECharts option={getColumnDistOption(selectedColumn)} style={{ height: '100%' }} />
                                            </div>
                                        </div>

                                        {/* Column Specific Stats */}
                                        <div className="grid grid-cols-2 gap-4 content-start">
                                            {selectedColumn.mean !== undefined ? (
                                                /* Numeric Stats */
                                                <>
                                                    <StatItem label="Mean" value={selectedColumn.mean.toFixed(2)} />
                                                    <StatItem label="Std Dev" value={selectedColumn.std?.toFixed(2)} />
                                                    <StatItem label="Median" value={selectedColumn.p50?.toFixed(2)} />
                                                    <StatItem label="Outliers" value={selectedColumn.outliers_count} />
                                                    <StatItem label="Skewness" value={selectedColumn.skewness?.toFixed(3)} />
                                                    <StatItem label="Kurtosis" value={selectedColumn.kurtosis?.toFixed(3)} />
                                                </>
                                            ) : (
                                                /* Categorical Stats */
                                                <>
                                                    <div className="col-span-2">
                                                        <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Top Values</h4>
                                                        <div className="space-y-2">
                                                            {selectedColumn.top_values?.map((v, i) => (
                                                                <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-slate-800/50">
                                                                    <span className="text-slate-300 truncate max-w-[150px]">{v.value}</span>
                                                                    <span className="text-white font-mono">{v.count}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* AI Deep Dive Insight */}
                        {report.insight && (
                            <motion.div 
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-8 border-t-4 border-cyan-500 bg-cyan-500/5"
                            >
                                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                    <Search className="w-5 h-5 text-cyan-400" />
                                    AI-Powered Dataset Deep Dive
                                </h3>
                                <div className="prose prose-invert max-w-none text-slate-300 leading-relaxed whitespace-pre-wrap">
                                    {report.insight.narrative}
                                </div>
                                <div className="mt-6 flex items-center gap-4 text-xs text-slate-500 font-mono">
                                    <span>Model: {report.insight.model_name}</span>
                                    <span>•</span>
                                    <span>Optimized for DataGuard platform</span>
                                </div>
                            </motion.div>
                        )}

                        {/* Correlation Heatmap */}
                        <div className="glass-card p-8">
                            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                                <Zap className="w-5 h-5 text-amber-400" />
                                Feature Correlation Heatmap
                            </h3>
                            <div className="h-[500px]">
                                <ReactECharts option={getCorrelationOption()} style={{ height: '100%' }} />
                            </div>
                        </div>

                        {/* Risks & Recommendations */}
                        {(report.top_risks.length > 0 || report.recommendations.length > 0) && (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="glass-card p-6 border-l-4 border-amber-500/50">
                                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                        <AlertCircle className="w-5 h-5 text-amber-500" />
                                        Potential Risks
                                    </h3>
                                    <ul className="space-y-3">
                                        {report.top_risks.map((risk, i) => (
                                            <li key={i} className="text-slate-300 flex items-start gap-2">
                                                <ChevronRight className="w-4 h-4 text-amber-500 mt-1 flex-shrink-0" />
                                                {risk}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="glass-card p-6 border-l-4 border-cyan-500/50">
                                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                        <CheckCircle2 className="w-5 h-5 text-cyan-500" />
                                        Recommendations
                                    </h3>
                                    <ul className="space-y-3">
                                        {report.recommendations.map((rec, i) => (
                                            <li key={i} className="text-slate-300 flex items-start gap-2">
                                                <ChevronRight className="w-4 h-4 text-cyan-500 mt-1 flex-shrink-0" />
                                                {rec}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}

function StatItem({ label, value }: { label: string, value: any }) {
    return (
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <p className="text-xs text-slate-500 uppercase font-bold mb-1">{label}</p>
            <p className="text-xl font-bold text-white">{value ?? '—'}</p>
        </div>
    )
}
