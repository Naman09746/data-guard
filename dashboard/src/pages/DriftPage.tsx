import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { 
    Activity, 
    Upload, 
    FileText, 
    AlertTriangle, 
    CheckCircle2, 
    ChevronRight, 
    BarChart3, 
    ArrowRightLeft,
    Zap,
    Download
} from 'lucide-react'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { analyzeDrift, type DriftResponse, type FeatureDrift } from '../api/client'

export default function DriftPage() {
    const [refFile, setRefFile] = useState<File | null>(null)
    const [currFile, setCurrFile] = useState<File | null>(null)
    const [report, setReport] = useState<DriftResponse | null>(null)
    const [selectedFeature, setSelectedFeature] = useState<FeatureDrift | null>(null)

    const mutation = useMutation({
        mutationFn: ({ ref, curr }: { ref: File; curr: File }) => 
            analyzeDrift(ref, curr),
        onSuccess: (data) => {
            setReport(data)
            setSelectedFeature(data.feature_drifts[0])
            toast.success('Drift analysis completed!')
        },
        onError: (error: any) => {
            toast.error(error.response?.data?.detail || 'Analysis failed')
        }
    })

    const handleRunAnalysis = () => {
        if (!refFile || !currFile) return
        mutation.mutate({ ref: refFile, curr: currFile })
    }

    const getDriftChartOption = (feature: FeatureDrift) => {
        return {
            tooltip: { trigger: 'axis' },
            legend: { data: ['Reference', 'Current'], textStyle: { color: '#94a3b8' } },
            grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
            xAxis: {
                type: 'category',
                data: Array.from({ length: feature.dist_ref.length }, (_, i) => `Bin ${i + 1}`),
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
            series: [
                {
                    name: 'Reference',
                    type: 'line',
                    smooth: true,
                    data: feature.dist_ref,
                    areaStyle: { opacity: 0.1 },
                    itemStyle: { color: '#6366f1' }
                },
                {
                    name: 'Current',
                    type: 'line',
                    smooth: true,
                    data: feature.dist_curr,
                    areaStyle: { opacity: 0.1 },
                    itemStyle: { color: '#f43f5e' }
                }
            ]
        }
    }

    return (
        <div className="space-y-8 pb-12">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Activity className="w-8 h-8 text-rose-500" />
                        ML Data Drift Analysis
                    </h1>
                    <p className="text-slate-400">Detect distribution shifts between training and production environments</p>
                </div>
            </div>

            {!report ? (
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-12 max-w-4xl mx-auto"
                >
                    <div className="grid grid-cols-2 gap-8 mb-8">
                        {/* Reference File */}
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest text-center">Reference Data (e.g. Training)</h3>
                            <label className="block p-8 border-2 border-dashed border-slate-700 rounded-2xl cursor-pointer hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all text-center">
                                <input type="file" accept=".csv" onChange={(e) => setRefFile(e.target.files?.[0] || null)} className="hidden" />
                                <Upload className={`w-8 h-8 mx-auto mb-2 ${refFile ? 'text-indigo-400' : 'text-slate-500'}`} />
                                <span className="text-slate-300 block truncate">{refFile ? refFile.name : 'Upload CSV'}</span>
                            </label>
                        </div>
                        {/* Current File */}
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest text-center">Current Data (e.g. Production)</h3>
                            <label className="block p-8 border-2 border-dashed border-slate-700 rounded-2xl cursor-pointer hover:border-rose-500/50 hover:bg-rose-500/5 transition-all text-center">
                                <input type="file" accept=".csv" onChange={(e) => setCurrFile(e.target.files?.[0] || null)} className="hidden" />
                                <Upload className={`w-8 h-8 mx-auto mb-2 ${currFile ? 'text-rose-400' : 'text-slate-500'}`} />
                                <span className="text-slate-300 block truncate">{currFile ? currFile.name : 'Upload CSV'}</span>
                            </label>
                        </div>
                    </div>

                    <button 
                        onClick={handleRunAnalysis}
                        disabled={!refFile || !currFile || mutation.isPending}
                        className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
                            !refFile || !currFile || mutation.isPending 
                            ? 'bg-slate-800 text-slate-500' 
                            : 'bg-gradient-to-r from-indigo-500 to-rose-600 text-white shadow-lg hover:scale-[1.01]'
                        }`}
                    >
                        {mutation.isPending ? 'Analyzing Distributions...' : 'Run Comparative Drift Analysis'}
                    </button>
                </motion.div>
            ) : (
                <div className="grid grid-cols-12 gap-8">
                    {/* Metrics Sidebar */}
                    <div className="col-span-12 lg:col-span-3 space-y-6">
                        <div className="glass-card p-6 text-center">
                            <h4 className="text-slate-400 text-sm mb-1 uppercase font-bold tracking-tighter">Overall Drift Score</h4>
                            <div className={`text-4xl font-black ${report.summary.drift_detected ? 'text-rose-500' : 'text-emerald-500'}`}>
                                {(report.summary.overall_drift_score * 100).toFixed(1)}%
                            </div>
                        </div>

                        <div className="glass-card p-4 max-h-[600px] overflow-y-auto custom-scrollbar">
                            <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 px-2">Analyzed Features</h3>
                            <div className="space-y-1">
                                {report.feature_drifts.map((f) => (
                                    <button
                                        key={f.name}
                                        onClick={() => setSelectedFeature(f)}
                                        className={`w-full p-3 rounded-xl flex items-center justify-between transition-all ${
                                            selectedFeature?.name === f.name 
                                            ? 'bg-rose-500/10 border border-rose-500/30 text-white' 
                                            : 'text-slate-400 hover:bg-slate-800/50'
                                        }`}
                                    >
                                        <span className="truncate max-w-[120px] font-medium">{f.name}</span>
                                        {f.is_drifted && <AlertTriangle className="w-4 h-4 text-rose-500" />}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Main Content */}
                    <div className="col-span-12 lg:col-span-9 space-y-8">
                        {/* Comparison Chart */}
                        <AnimatePresence mode="wait">
                            {selectedFeature && (
                                <motion.div 
                                    key={selectedFeature.name}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="glass-card p-8"
                                >
                                    <div className="flex items-center justify-between mb-8">
                                        <div className="flex items-center gap-4">
                                            <div className={`p-3 rounded-2xl ${selectedFeature.is_drifted ? 'bg-rose-500/10' : 'bg-emerald-500/10'}`}>
                                                <ArrowRightLeft className={`w-6 h-6 ${selectedFeature.is_drifted ? 'text-rose-400' : 'text-emerald-400'}`} />
                                            </div>
                                            <div>
                                                <h2 className="text-2xl font-bold text-white">{selectedFeature.name}</h2>
                                                <p className="text-slate-400">PSI Score: {selectedFeature.score.toFixed(4)} • P-Value: {selectedFeature.p_value.toFixed(4)}</p>
                                            </div>
                                        </div>
                                        {selectedFeature.is_drifted ? (
                                            <div className="px-4 py-2 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm font-bold animate-pulse">
                                                DRIFT DETECTED
                                            </div>
                                        ) : (
                                            <div className="px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-bold">
                                                STABLE
                                            </div>
                                        )}
                                    </div>

                                    <div className="h-80 w-full">
                                        <ReactECharts option={getDriftChartOption(selectedFeature)} style={{ height: '100%' }} />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Summary Details */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="glass-card p-6">
                                <h3 className="text-lg font-bold text-white mb-4">Drifted Features ({report.summary.drifted_features_count})</h3>
                                <div className="flex flex-wrap gap-2">
                                    {report.summary.drifted_features.map((f) => (
                                        <span key={f} className="px-3 py-1 rounded-lg bg-rose-500/10 text-rose-400 text-sm border border-rose-500/20">
                                            {f}
                                        </span>
                                    ))}
                                    {report.summary.drifted_features.length === 0 && (
                                        <span className="text-slate-500 italic text-sm">No drift detected in numeric features</span>
                                    )}
                                </div>
                            </div>
                            <div className="glass-card p-6">
                                <h3 className="text-lg font-bold text-white mb-4">Recommended Actions</h3>
                                <ul className="space-y-3">
                                    {report.summary.drift_detected ? (
                                        <>
                                            <li className="flex items-start gap-2 text-slate-300 text-sm">
                                                <Zap className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                                                Retrain model using the most recent data window.
                                            </li>
                                            <li className="flex items-start gap-2 text-slate-300 text-sm">
                                                <Zap className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                                                Investigate feature engineering logic for the drifted columns.
                                            </li>
                                        </>
                                    ) : (
                                        <li className="flex items-start gap-2 text-slate-300 text-sm">
                                            <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                                            Data remains stable. No immediate retraining required.
                                        </li>
                                    )}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
