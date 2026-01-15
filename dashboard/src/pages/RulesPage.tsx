import { useState } from 'react'
import {
    Settings,
    Plus,
    Trash2,
    Save,
    Mail,
    Phone,
    CreditCard,
    Calendar,
    Hash,
    Type,
    AlertTriangle,
} from 'lucide-react'

interface ValidationRule {
    id: string
    name: string
    type: string
    column: string
    parameters: Record<string, any>
    enabled: boolean
}

const ruleTypes = [
    { value: 'email', label: 'Email Format', icon: Mail },
    { value: 'phone', label: 'Phone Number', icon: Phone },
    { value: 'credit_card', label: 'Credit Card', icon: CreditCard },
    { value: 'date', label: 'Date Format', icon: Calendar },
    { value: 'range', label: 'Numeric Range', icon: Hash },
    { value: 'pattern', label: 'Regex Pattern', icon: Type },
]

const defaultRules: ValidationRule[] = [
    { id: '1', name: 'Email Validation', type: 'email', column: 'email', parameters: {}, enabled: true },
    { id: '2', name: 'Age Range', type: 'range', column: 'age', parameters: { min: 0, max: 120 }, enabled: true },
    { id: '3', name: 'Phone Format', type: 'phone', column: 'phone', parameters: { format: 'international' }, enabled: false },
]

export default function RulesPage() {
    const [rules, setRules] = useState<ValidationRule[]>(defaultRules)
    const [showAddModal, setShowAddModal] = useState(false)
    const [newRule, setNewRule] = useState({
        name: '',
        type: 'email',
        column: '',
        parameters: {} as Record<string, any>,
    })

    const toggleRule = (id: string) => {
        setRules(rules.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r))
    }

    const deleteRule = (id: string) => {
        setRules(rules.filter(r => r.id !== id))
    }

    const addRule = () => {
        if (!newRule.name || !newRule.column) return

        setRules([
            ...rules,
            {
                ...newRule,
                id: Date.now().toString(),
                enabled: true,
            },
        ])
        setShowAddModal(false)
        setNewRule({ name: '', type: 'email', column: '', parameters: {} })
    }

    const getRuleIcon = (type: string) => {
        const rule = ruleTypes.find(r => r.value === type)
        return rule?.icon || AlertTriangle
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Custom Rules</h1>
                    <p className="text-slate-400">
                        Configure custom validation rules for your data quality checks
                    </p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium flex items-center gap-2 hover:opacity-90 transition-all"
                >
                    <Plus className="w-5 h-5" />
                    Add Rule
                </button>
            </div>

            {/* Rule Types */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {ruleTypes.map((type) => {
                    const Icon = type.icon
                    const count = rules.filter(r => r.type === type.value && r.enabled).length
                    return (
                        <div
                            key={type.value}
                            className="glass-card p-4 flex flex-col items-center gap-2 hover-lift cursor-pointer"
                        >
                            <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center">
                                <Icon className="w-5 h-5 text-indigo-400" />
                            </div>
                            <span className="text-sm text-white font-medium">{type.label}</span>
                            <span className="text-xs text-slate-400">{count} active</span>
                        </div>
                    )
                })}
            </div>

            {/* Rules List */}
            <div className="glass-card overflow-hidden">
                <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Settings className="w-5 h-5 text-slate-400" />
                        Active Rules
                    </h3>
                    <span className="text-sm text-slate-400">
                        {rules.filter(r => r.enabled).length} of {rules.length} enabled
                    </span>
                </div>

                <div className="divide-y divide-slate-700/50">
                    {rules.map((rule) => {
                        const Icon = getRuleIcon(rule.type)
                        return (
                            <div
                                key={rule.id}
                                className={`p-4 flex items-center justify-between transition-colors ${rule.enabled ? '' : 'opacity-50'
                                    }`}
                            >
                                <div className="flex items-center gap-4">
                                    <div
                                        className={`w-10 h-10 rounded-xl flex items-center justify-center ${rule.enabled ? 'bg-indigo-500/20' : 'bg-slate-700/50'
                                            }`}
                                    >
                                        <Icon className={`w-5 h-5 ${rule.enabled ? 'text-indigo-400' : 'text-slate-500'}`} />
                                    </div>
                                    <div>
                                        <p className="text-white font-medium">{rule.name}</p>
                                        <p className="text-sm text-slate-400">
                                            Column: <code className="text-indigo-400">{rule.column}</code>
                                            {Object.keys(rule.parameters).length > 0 && (
                                                <span className="ml-2">
                                                    • {JSON.stringify(rule.parameters)}
                                                </span>
                                            )}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={() => toggleRule(rule.id)}
                                        className={`w-12 h-6 rounded-full transition-colors ${rule.enabled ? 'bg-indigo-500' : 'bg-slate-600'
                                            } relative`}
                                    >
                                        <div
                                            className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${rule.enabled ? 'translate-x-6' : 'translate-x-0.5'
                                                }`}
                                        />
                                    </button>
                                    <button
                                        onClick={() => deleteRule(rule.id)}
                                        className="p-2 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Add Rule Modal */}
            {showAddModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="glass-card p-6 w-full max-w-md space-y-4">
                        <h3 className="text-xl font-bold text-white">Add Custom Rule</h3>

                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Rule Name</label>
                            <input
                                type="text"
                                value={newRule.name}
                                onChange={(e) => setNewRule({ ...newRule, name: e.target.value })}
                                placeholder="e.g., Email Validation"
                                className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                            />
                        </div>

                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Rule Type</label>
                            <select
                                value={newRule.type}
                                onChange={(e) => setNewRule({ ...newRule, type: e.target.value })}
                                className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white focus:outline-none focus:border-indigo-500"
                            >
                                {ruleTypes.map((type) => (
                                    <option key={type.value} value={type.value}>
                                        {type.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Column Name</label>
                            <input
                                type="text"
                                value={newRule.column}
                                onChange={(e) => setNewRule({ ...newRule, column: e.target.value })}
                                placeholder="e.g., email, phone, age"
                                className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                            />
                        </div>

                        {newRule.type === 'range' && (
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm text-slate-400 mb-2">Min</label>
                                    <input
                                        type="number"
                                        onChange={(e) =>
                                            setNewRule({
                                                ...newRule,
                                                parameters: { ...newRule.parameters, min: Number(e.target.value) },
                                            })
                                        }
                                        className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white focus:outline-none focus:border-indigo-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm text-slate-400 mb-2">Max</label>
                                    <input
                                        type="number"
                                        onChange={(e) =>
                                            setNewRule({
                                                ...newRule,
                                                parameters: { ...newRule.parameters, max: Number(e.target.value) },
                                            })
                                        }
                                        className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white focus:outline-none focus:border-indigo-500"
                                    />
                                </div>
                            </div>
                        )}

                        {newRule.type === 'pattern' && (
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Regex Pattern</label>
                                <input
                                    type="text"
                                    onChange={(e) =>
                                        setNewRule({
                                            ...newRule,
                                            parameters: { pattern: e.target.value },
                                        })
                                    }
                                    placeholder="e.g., ^[A-Z]{2}[0-9]{4}$"
                                    className="w-full px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-600 text-white font-mono placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                                />
                            </div>
                        )}

                        <div className="flex gap-3 pt-4">
                            <button
                                onClick={() => setShowAddModal(false)}
                                className="flex-1 px-4 py-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700/50 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={addRule}
                                className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium flex items-center justify-center gap-2 hover:opacity-90"
                            >
                                <Save className="w-4 h-4" />
                                Save Rule
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
