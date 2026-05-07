import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation } from '@tanstack/react-query'
import {
    Upload,
    Shield,
    CheckCircle,
    AlertCircle,
    AlertTriangle,
    Loader2,
    ChevronDown,
    ChevronRight,
    Zap,
    Network,
    Activity,
    Search,
    Maximize2
} from 'lucide-react'
import ReactECharts from 'echarts-for-react'
import { motion, AnimatePresence } from 'framer-motion'
import { detectLeakage, getLeakageNetwork, type LeakageResponse, type NetworkGraph } from '../api/client'
import { addValidationRecord } from '../store/validationHistory'

export default function LeakagePage() {
    const [trainFile, setTrainFile] = useState<File | null>(null)
    const [testFile, setTestFile] = useState<File | null>(null)
    const [targetColumn, setTargetColumn] = useState('')
    const [result, setResult] = useState<LeakageResponse | null>(null)
    const [networkData, setNetworkData] = useState<NetworkGraph | null>(null)
    const [expandedDetector, setExpandedDetector] = useState<string | null>(null)
    const [activeTab, setActiveTab] = useState<'issues' | 'network' | 'heatmap'>('issues')

    const mutation = useMutation({
        mutationFn: async () => {
            const report = await detectLeakage(trainFile!, testFile, targetColumn || undefined)
            // If target is specified, also fetch the network graph
            if (targetColumn) {
                const network = await getLeakageNetwork(trainFile!, targetColumn)
                setNetworkData(network)
            }
            return report
        },
        onSuccess: (data) => {
            setResult(data)
            addValidationRecord({
                type: 'leakage',
                fileName: trainFile?.name || 'Training data',
                status: data.is_clean ? 'clean' : 'detected',
                totalIssues: data.total_issues,
                duration: data.duration_seconds,
                rows: data.summary?.train_rows,
            })
        },
    })

    const onDropTrain = useCallback((files: File[]) => {
        if (files.length > 0) setTrainFile(files[0])
    }, [])

    const onDropTest = useCallback((files: File[]) => {
        if (files.length > 0) setTestFile(files[0])
    }, [])

    const trainDropzone = useDropzone({
        onDrop: onDropTrain,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const testDropzone = useDropzone({
        onDrop: onDropTest,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const getStatusIcon = (status: string) => {
        if (status === 'clean') return <CheckCircle className="w-5 h-5 text-emerald-400" />
        if (status === 'detected') return <AlertCircle className="w-5 h-5 text-red-400" />
        return <AlertTriangle className="w-5 h-5 text-amber-400" />
    }

    const getNetworkOption = (data: NetworkGraph) => {
        return {
            tooltip: {
                formatter: (params: any) => {
                    if (params.dataType === 'node') {
                        return `<b>${params.name}</b><br/>Risk Score: ${(params.data.risk_score * 100).toFixed(1)}%`
                    }
                    return `Correlation: ${(params.data.value * 100).toFixed(1)}%`
                }
            },
            series: [
                {
                    type: 'graph',
                    layout: 'force',
                    animation: true,
                    draggable: true,
                    data: data.nodes.map(node => ({
                        ...node,
                        symbolSize: node.is_target ? 50 : 25 + (node.risk_score * 30),
                        itemStyle: {
                            color: node.is_target ? '#6366f1' : (node.risk_score > 0.7 ? '#f43f5e' : '#94a3b8'),
                            borderColor: node.risk_score > 0.7 ? '#f43f5e' : '#1e293b',
                            borderWidth: node.risk_score > 0.7 ? 2 : 0,
                            shadowBlur: node.risk_score > 0.7 ? 10 : 0,
                            shadowColor: node.risk_score > 0.7 ? '#f43f5e' : 'transparent'
                        },
                        label: {
                            show: true,
                            position: 'right',
                            color: '#e2e8f0',
                            fontSize: 10
                        }
                    })),
                    links: data.links,
                    force: {
                        repulsion: 200,
                        edgeLength: 100,
                        gravity: 0.1
                    },
                    lineStyle: {
                        opacity: 0.4,
                        curveness: 0.1,
                        color: '#475569'
                    },
                    emphasis: {
                        focus: 'adjacency',
                        lineStyle: {
                            width: 3,
                            opacity: 1,
                            color: '#6366f1'
                        }
                    }
                }
            ]
        }
    }

    return (
        <div className="space-y-8 pb-12">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Shield className="w-8 h-8 text-indigo-500" />
                        ML Leakage Intelligence
                    </h1>
                    <p className="text-slate-400">Expose hidden predictive "cheating" and data contamination</p>
                </div>
            </div>

            {/* Upload Areas */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div
                    {...trainDropzone.getRootProps()}
                    className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${trainDropzone.isDragActive
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : trainFile
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-slate-600 hover:border-indigo-500/50'
                        }`}
                >
                    <input {...trainDropzone.getInputProps()} />
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Upload className={`w-10 h-10 ${trainFile ? 'text-emerald-400' : 'text-slate-400'}`} />
                        <div>
                            <p className="text-white font-medium">Training Data (Reference)</p>
                            {trainFile ? (
                                <p className="text-sm text-emerald-400">{trainFile.name}</p>
                            ) : (
                                <p className="text-sm text-slate-400">Drop CSV here</p>
                            )}
                        </div>
                    </div>
                </div>

                <div
                    {...testDropzone.getRootProps()}
                    className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${testDropzone.isDragActive
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : testFile
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-slate-600 hover:border-indigo-500/50'
                        }`}
                >
                    <input {...testDropzone.getInputProps()} />
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Upload className={`w-10 h-10 ${testFile ? 'text-emerald-400' : 'text-slate-400'}`} />
                        <div>
                            <p className="text-white font-medium">Holdout / Test Set</p>
                            {testFile ? (
                                <p className="text-sm text-emerald-400">{testFile.name}</p>
                            ) : (
                                <p className="text-sm text-slate-400">Optional comparison set</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div className="flex items-end gap-6 glass-card p-6">
                <div className="flex-1">
                    <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 px-1">Target Prediction Column</label>
                    <div className="relative">
                        <Search className="absolute left-4 top-3.5 w-5 h-5 text-slate-500" />
                        <input
                            type="text"
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            placeholder="Identify the column you are predicting (e.g. 'label')"
                            className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-slate-900/50 border border-slate-700 text-white focus:border-indigo-500 transition-all outline-none"
                        />
                    </div>
                </div>
                <button
                    onClick={() => mutation.mutate()}
                    disabled={!trainFile || mutation.isPending || !targetColumn}
                    className="h-12 px-8 rounded-xl bg-gradient-to-r from-indigo-500 to-rose-600 text-white font-bold flex items-center gap-2 hover:scale-[1.02] disabled:opacity-50 transition-all shadow-lg shadow-indigo-500/20"
                >
                    {mutation.isPending ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
                    Analyze Leakage
                </button>
            </div>

            {/* Results Section */}
            {result && (
                <div className="space-y-6">
                    {/* Summary Hero */}
                    <motion.div 
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`glass-card p-8 border-l-8 ${result.is_clean ? 'border-emerald-500 bg-emerald-500/5' : 'border-rose-500 bg-rose-500/5'}`}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-6">
                                <div className={`p-4 rounded-2xl ${result.is_clean ? 'bg-emerald-500/10' : 'bg-rose-500/10'}`}>
                                    <Shield className={`w-10 h-10 ${result.is_clean ? 'text-emerald-400' : 'text-rose-400'}`} />
                                </div>
                                <div>
                                    <h2 className="text-3xl font-black text-white">{result.is_clean ? 'SCAN CLEAN' : 'LEAKAGE DETECTED'}</h2>
                                    <p className="text-slate-400 font-medium">
                                        {result.total_issues} security risks identified • {result.duration_seconds.toFixed(2)}s analysis time
                                    </p>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-tighter mb-1">Global Risk Score</div>
                                <div className={`text-4xl font-black ${result.is_clean ? 'text-emerald-500' : 'text-rose-500'}`}>
                                    {result.is_clean ? '0' : '84'} <span className="text-xl">/ 100</span>
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* View Tabs */}
                    <div className="flex items-center gap-4 border-b border-slate-800 pb-4">
                        <button 
                            onClick={() => setActiveTab('issues')}
                            className={`px-4 py-2 rounded-lg font-bold transition-all ${activeTab === 'issues' ? 'bg-indigo-500 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            Issues List
                        </button>
                        <button 
                            onClick={() => setActiveTab('network')}
                            className={`px-4 py-2 rounded-lg font-bold transition-all ${activeTab === 'network' ? 'bg-indigo-500 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            Force Correlation Network
                        </button>
                    </div>

                    <AnimatePresence mode="wait">
                        {activeTab === 'issues' ? (
                            <motion.div 
                                key="issues"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className="space-y-4"
                            >
                                {result.results.map((detector: any, index: number) => (
                                    <div key={index} className="glass-card overflow-hidden">
                                        <button
                                            onClick={() => setExpandedDetector(expandedDetector === detector.detector_name ? null : detector.detector_name)}
                                            className="w-full p-6 flex items-center justify-between hover:bg-slate-700/20 transition-colors"
                                        >
                                            <div className="flex items-center gap-4">
                                                {getStatusIcon(detector.status)}
                                                <span className="text-lg font-bold text-white">{detector.detector_name}</span>
                                            </div>
                                            {expandedDetector === detector.detector_name ? <ChevronDown className="text-slate-500" /> : <ChevronRight className="text-slate-500" />}
                                        </button>
                                        
                                        {expandedDetector === detector.detector_name && detector.issues?.length > 0 && (
                                            <div className="px-6 pb-6 space-y-4">
                                                {detector.issues.map((issue: any, i: number) => (
                                                    <div key={i} className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800">
                                                        <div className="flex items-start gap-4">
                                                            <div className="p-2 rounded-lg bg-rose-500/10"><AlertTriangle className="w-5 h-5 text-rose-500" /></div>
                                                            <div className="space-y-3">
                                                                <p className="text-white font-medium">{issue.message}</p>
                                                                <div className="p-3 rounded-xl bg-indigo-500/5 border border-indigo-500/20 text-indigo-300 text-sm italic">
                                                                    💡 Recommendation: {issue.recommendation}
                                                                </div>
                                                                {issue.affected_features?.length > 0 && (
                                                                    <div className="flex gap-2 flex-wrap pt-2">
                                                                        {issue.affected_features.map((f: string) => (
                                                                            <span key={f} className="px-3 py-1 rounded-lg bg-slate-800 text-slate-400 text-xs font-mono border border-slate-700">
                                                                                {f}
                                                                            </span>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </motion.div>
                        ) : (
                            <motion.div 
                                key="network"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className="glass-card p-8 h-[600px] relative"
                            >
                                <div className="absolute top-8 left-8 z-10">
                                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                                        <Network className="w-5 h-5 text-indigo-400" />
                                        Interactive Correlation Map
                                    </h3>
                                    <p className="text-slate-400 text-sm">Nodes size/color indicate risk level. Edges show correlation strength.</p>
                                </div>
                                <div className="absolute top-8 right-8 z-10 flex gap-4">
                                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-indigo-500"></div><span className="text-xs text-slate-400">Target</span></div>
                                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-rose-500"></div><span className="text-xs text-slate-400">High Risk</span></div>
                                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-slate-500"></div><span className="text-xs text-slate-400">Normal</span></div>
                                </div>
                                {networkData ? (
                                    <ReactECharts option={getNetworkOption(networkData)} style={{ height: '100%', width: '100%' }} />
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-slate-500 italic">
                                        <Activity className="w-12 h-12 mb-4 opacity-20" />
                                        Please provide a target column to generate network graph
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </div>
    )
}
