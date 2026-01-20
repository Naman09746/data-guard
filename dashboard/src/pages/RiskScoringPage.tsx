import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation } from '@tanstack/react-query'
import {
    Upload,
    Brain,
    AlertTriangle,
    CheckCircle,
    Loader2,
    BarChart3,
    Lightbulb,
    AlertCircle,
} from 'lucide-react'
import { getRiskScores, type RiskScoresResponse, type FeatureRiskScore } from '../api/client'

export default function RiskScoringPage() {
    const [file, setFile] = useState<File | null>(null)
    const [targetColumn, setTargetColumn] = useState('')
    const [result, setResult] = useState<RiskScoresResponse | null>(null)
    const [selectedFeature, setSelectedFeature] = useState<FeatureRiskScore | null>(null)

    const mutation = useMutation({
        mutationFn: () => getRiskScores(file!, targetColumn),
        onSuccess: (data) => setResult(data),
    })

    const onDrop = useCallback((files: File[]) => {
        if (files.length > 0) setFile(files[0])
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const getRiskColor = (level: string) => {
        if (level === 'high') return 'text-red-400'
        if (level === 'medium') return 'text-amber-400'
        return 'text-emerald-400'
    }

    const getRiskBg = (level: string) => {
        if (level === 'high') return 'bg-red-500/20 border-red-500/30'
        if (level === 'medium') return 'bg-amber-500/20 border-amber-500/30'
        return 'bg-emerald-500/20 border-emerald-500/30'
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                    <Brain className="w-8 h-8 text-indigo-400" />
                    ML-Based Risk Scoring
                </h1>
                <p className="text-slate-400">
                    Get probability-based risk scores for each feature using machine learning
                </p>
            </div>

            {/* Upload Area */}
            <div
                {...getRootProps()}
                className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${isDragActive
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : file
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-slate-600 hover:border-indigo-500/50'
                    }`}
            >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-3 text-center">
                    <Upload className={`w-10 h-10 ${file ? 'text-emerald-400' : 'text-slate-400'}`} />
                    <div>
                        <p className="text-white font-medium">Upload Dataset</p>
                        {file ? (
                            <p className="text-sm text-emerald-400">{file.name}</p>
                        ) : (
                            <p className="text-sm text-slate-400">Drop or click to upload CSV</p>
                        )}
                    </div>
                </div>
            </div>

            {/* Target Column & Run */}
            <div className="flex items-center gap-4">
                <div className="flex-1">
                    <label className="block text-sm text-slate-400 mb-2">Target Column (required)</label>
                    <input
                        type="text"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        placeholder="e.g., target, churned, is_fraud"
                        className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-600 text-white placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                    />
                </div>
                <div className="pt-7">
                    <button
                        onClick={() => mutation.mutate()}
                        disabled={!file || !targetColumn || mutation.isPending}
                        className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium flex items-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                        {mutation.isPending ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <BarChart3 className="w-5 h-5" />
                        )}
                        Compute Risk Scores
                    </button>
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className="space-y-6">
                    {/* Summary */}
                    <div className={`glass-card p-6 border ${getRiskBg(result.overall_risk)}`}>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${result.overall_risk === 'high' ? 'bg-red-500/20' :
                                        result.overall_risk === 'medium' ? 'bg-amber-500/20' : 'bg-emerald-500/20'
                                    }`}>
                                    {result.overall_risk === 'high' ? (
                                        <AlertTriangle className="w-6 h-6 text-red-400" />
                                    ) : result.overall_risk === 'medium' ? (
                                        <AlertCircle className="w-6 h-6 text-amber-400" />
                                    ) : (
                                        <CheckCircle className="w-6 h-6 text-emerald-400" />
                                    )}
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold text-white">
                                        Overall Risk: <span className={getRiskColor(result.overall_risk)}>{result.overall_risk.toUpperCase()}</span>
                                    </h3>
                                    <p className="text-slate-400">
                                        {result.high_risk_count} high-risk • {result.medium_risk_count} medium-risk • {result.duration_seconds.toFixed(2)}s
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Feature Scores Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Feature List */}
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Feature Risk Scores</h3>
                            <div className="space-y-3 max-h-[500px] overflow-y-auto">
                                {result.feature_scores
                                    .sort((a, b) => b.risk_score - a.risk_score)
                                    .map((score) => (
                                        <button
                                            key={score.feature_name}
                                            onClick={() => setSelectedFeature(score)}
                                            className={`w-full p-4 rounded-lg text-left transition-all hover:bg-slate-700/50 border ${selectedFeature?.feature_name === score.feature_name
                                                    ? 'border-indigo-500 bg-indigo-500/10'
                                                    : 'border-transparent'
                                                }`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="text-white font-medium">{score.feature_name}</span>
                                                <span className={`font-bold ${getRiskColor(score.risk_level)}`}>
                                                    {score.risk_percentage}%
                                                </span>
                                            </div>
                                            <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full transition-all ${score.risk_level === 'high' ? 'bg-red-500' :
                                                            score.risk_level === 'medium' ? 'bg-amber-500' : 'bg-emerald-500'
                                                        }`}
                                                    style={{ width: `${score.risk_percentage}%` }}
                                                />
                                            </div>
                                        </button>
                                    ))}
                            </div>
                        </div>

                        {/* Feature Details */}
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Feature Details</h3>
                            {selectedFeature ? (
                                <div className="space-y-6">
                                    <div className={`p-4 rounded-lg border ${getRiskBg(selectedFeature.risk_level)}`}>
                                        <h4 className="font-bold text-white text-lg">{selectedFeature.feature_name}</h4>
                                        <p className={`text-2xl font-bold ${getRiskColor(selectedFeature.risk_level)}`}>
                                            {selectedFeature.risk_percentage}% Risk
                                        </p>
                                    </div>

                                    <div>
                                        <h5 className="text-sm text-slate-400 mb-2">Contributing Factors</h5>
                                        <div className="space-y-2">
                                            {Object.entries(selectedFeature.contributing_factors).map(([key, value]) => (
                                                <div key={key} className="flex justify-between p-2 bg-slate-800/50 rounded-lg">
                                                    <span className="text-slate-300 text-sm">{key.replace(/_/g, ' ')}</span>
                                                    <span className="text-white font-mono text-sm">{value.toFixed(4)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {selectedFeature.recommendations.length > 0 && (
                                        <div>
                                            <h5 className="text-sm text-slate-400 mb-2 flex items-center gap-2">
                                                <Lightbulb className="w-4 h-4" />
                                                Recommendations
                                            </h5>
                                            <ul className="space-y-2">
                                                {selectedFeature.recommendations.map((rec, idx) => (
                                                    <li key={idx} className="p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-lg text-sm text-indigo-300">
                                                        {rec}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                                    <BarChart3 className="w-12 h-12 mb-4 opacity-50" />
                                    <p>Select a feature to view details</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
