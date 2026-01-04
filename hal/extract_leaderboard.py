import pandas as pd
from bs4 import BeautifulSoup
import io

html_content = """

            <div class="flex justify-between items-center mb-6">
                <h3 class="text-3xl font-semibold">CORE-Bench Hard Leaderboard</h3>
                
            </div>
            <div class="mb-4 p-4 bg-blue-50 rounded-lg">
                <p class="text-sm text-gray-700">
                    Costs are currently calculated without accounting for caching benefits.
                </p>
            </div>
            <div class="mb-4 p-4 bg-green-50 rounded-lg">
                <p class="text-sm text-gray-700">
                    <strong>Update:</strong> Running Opus 4.5 with an updated scaffold that uses Claude Code drastically outperforms the CORE-Agent scaffold we used, especially after fixing a few grading errors via manual scoring. We have now declared <a href="https://x.com/sayashk/status/1996334941832089732?t=1tFle-jfHsDHFEOSjyo9mg&amp;s=19" class="text-green-600 hover:text-green-800 underline" target="_blank" rel="noopener noreferrer">CORE-Bench solved</a>.
                </p>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Scaffold</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider group relative">
                                <span class="cursor-help inline-flex items-center">
                                    Primary Model
                                    <svg class="w-4 h-4 ml-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                </span>
                                <div class="invisible group-hover:visible absolute z-50 w-64 p-4 text-sm bg-white rounded-lg shadow-lg border border-gray-200 -translate-x-1/2 left-1/2 top-full mt-1">
                                    <div class="absolute w-4 h-4 bg-white transform rotate-45 -translate-x-1/2 left-1/2 -top-2 border-t border-l border-gray-200"></div>
                                    <div class="flex items-center mb-2">
                                        <svg class="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 4.5v1.5m3-1.5v1.5m3-1.5v1.5M4.5 9h1.5m-1.5 3h1.5m-1.5 3h1.5M18 9h1.5m-1.5 3h1.5m-1.5 3h1.5M9 18v1.5m3-1.5v1.5m3-1.5v1.5M7.5 6h9A1.5 1.5 0 0118 7.5v9a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 016 16.5v-9A1.5 1.5 0 017.5 6z"></path>
                                        </svg>
                                        <span class="font-medium text-gray-900">Primary Model</span>
                                    </div>
                                    <p class="text-gray-600 leading-relaxed">This is the primary model used by the agent. In some cases, an embedding model is used for RAG, or a secondary model like GPT-4o for image processing.
                                        <span class="font-semibold">Note:</span> For non-OpenAI reasoning models, the reasoning token budget is set at 1,024 (low), 2,048 (medium), and 4,096 (high).
                                    </p>
                                </div>
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider group relative">
                                <span class="cursor-help inline-flex items-center">
                                    Verified
                                    <svg class="w-4 h-4 ml-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                </span>
                                <div class="invisible group-hover:visible absolute z-50 w-64 p-4 text-sm bg-white rounded-lg shadow-lg border border-gray-200 -translate-x-1/2 left-1/2 top-full mt-1">
                                    <div class="absolute w-4 h-4 bg-white transform rotate-45 -translate-x-1/2 left-1/2 -top-2 border-t border-l border-gray-200"></div>
                                    <div class="flex items-center mb-2">
                                        <svg class="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                        </svg>
                                        <span class="font-medium text-gray-900">Verified Results</span>
                                    </div>
                                    <p class="text-gray-600 leading-relaxed">Results have been reproduced by the HAL team</p>
                                </div>
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider group relative">
                                <span class="cursor-help inline-flex items-center">
                                    Accuracy
                                    <svg class="w-4 h-4 ml-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                </span>
                                <div class="invisible group-hover:visible absolute z-50 w-64 p-4 text-sm bg-white rounded-lg shadow-lg border border-gray-200 -translate-x-1/2 left-1/2 top-full mt-1">
                                    <div class="absolute w-4 h-4 bg-white transform rotate-45 -translate-x-1/2 left-1/2 -top-2 border-t border-l border-gray-200"></div>
                                    <div class="flex items-center mb-2">
                                        <svg class="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                                        </svg>
                                        <span class="font-medium text-gray-900">Accuracy</span>
                                    </div>
                                    <p class="text-gray-600 leading-relaxed">Confidence intervals show the min-max values across runs for those agents where multiple runs are available</p>
                                </div>
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider group relative">
                                <span class="cursor-help inline-flex items-center">
                                    Cost (USD)
                                    <svg class="w-4 h-4 ml-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                </span>
                                <div class="invisible group-hover:visible absolute z-50 w-64 p-4 text-sm bg-white rounded-lg shadow-lg border border-gray-200 -translate-x-1/2 left-1/2 top-full mt-1">
                                    <div class="absolute w-4 h-4 bg-white transform rotate-45 -translate-x-1/2 left-1/2 -top-2 border-t border-l border-gray-200"></div>
                                    <div class="flex items-center mb-2">
                                        <svg class="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                        </svg>
                                        <span class="font-medium text-gray-900">Total Cost</span>
                                    </div>
                                    <p class="text-gray-600 leading-relaxed">Total API cost for running the agent on all tasks. Confidence intervals show the min-max values across runs for those agents where multiple runs are available</p>
                                </div>
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider group relative">
                                <span class="cursor-help inline-flex items-center">
                                    Runs
                                    <svg class="w-4 h-4 ml-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                </span>
                                <div class="invisible group-hover:visible absolute z-50 w-64 p-4 text-sm bg-white rounded-lg shadow-lg border border-gray-200 -translate-x-1/2 left-1/2 top-full mt-1">
                                    <div class="absolute w-4 h-4 bg-white transform rotate-45 -translate-x-1/2 left-1/2 -top-2 border-t border-l border-gray-200"></div>
                                    <div class="flex items-center mb-2">
                                        <svg class="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                        </svg>
                                        <span class="font-medium text-gray-900">Number of Runs</span>
                                    </div>
                                    <p class="text-gray-600 leading-relaxed">The number of runs for this agent submitted to the leaderboard. To submit multiple evaluations, rerun the same agent and set the same agent name</p>
                                </div>
                            </th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Traces</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        
                        <tr class="hover:bg-gray-50 bg-blue-50/40">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">1</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                        <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Pareto optimal</span>
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.1%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.1 (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    51.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $412.42
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754492675_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 bg-blue-50/40">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">2</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                        <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Pareto optimal</span>
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204.5%20High%20(September%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4.5 High (September 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    44.44%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $92.34
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet45high_1759330487_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">3</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.5%20High%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.5 High (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    42.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $152.66
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_opus_45_high_1764027725_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">4</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.5%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.5 (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    42.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $168.99
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_opus_45_1764027531_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">5</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.1%20High%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.1 High (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    42.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $509.95
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754539779_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">6</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%203%20Pro%20Preview%20High%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 3 Pro Preview High (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    40.00%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $86.60
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_1763842609_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">7</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude-3.7%20Sonnet%20High%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude-3.7 Sonnet High (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    37.78%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $66.15
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentsonnet_37_high_1755663010_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">8</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204.5%20(September%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4.5 (September 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    37.78%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $97.15
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet45_1759329435_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 bg-blue-50/40">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">9</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                        <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Pareto optimal</span>
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o4-mini%20High%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o4-mini High (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    35.56%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $45.37
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento4minihigh_1755580383_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">10</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude-3.7%20Sonnet%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude-3.7 Sonnet (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    35.56%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $73.04
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922181_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">11</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%203%20Pro%20Preview%20High%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 3 Pro Preview High (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    35.56%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $101.27
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1763837773_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">12</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.1%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.1 (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    35.56%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $375.11
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754443772_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">13</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204.5%20(September%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4.5 (September 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    33.33%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $85.19
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_halgeneralistagentclaudesonnet45_1759433359_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">14</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204%20High%20(May%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4 High (May 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    33.33%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $100.48
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet4high_1755814601_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">15</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-4.1%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-4.1 (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    33.33%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $107.36
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744752123_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">16</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.5%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.5 (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    33.33%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $127.41
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_opus_45_1764046559_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">17</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.1%20High%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.1 High (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    33.33%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $358.47
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754569694_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">18</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude-3.7%20Sonnet%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude-3.7 Sonnet (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    31.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $56.64
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentsonnet_37_1755652380_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">19</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Opus%204.5%20High%20(November%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Opus 4.5 High (November 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    31.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $112.38
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_opus_45_high_1764046628_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">20</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204%20(May%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4 (May 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    28.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $50.27
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet4_1755796611_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">21</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Sonnet%204.5%20High%20(September%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Sonnet 4.5 High (September 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    28.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $87.77
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_halgeneralistagentclaudesonnet45high_1759423572_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">22</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-5%20Medium%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-5 Medium (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    26.67%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $31.76
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754599494_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">23</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o4-mini%20High%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o4-mini High (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    26.67%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $61.35
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745075792_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">24</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude-3.7%20Sonnet%20High%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude-3.7 Sonnet High (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    24.44%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $72.47
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745258007_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">25</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o3%20Medium%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o3 Medium (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    24.44%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $120.47
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745118876_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">26</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-4.1%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-4.1 (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    22.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $58.32
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgpt41_1755644685_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">27</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o3%20Medium%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o3 Medium (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    22.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $88.34
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento3medium_1755626315_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">28</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%202.5%20Pro%20Preview%20(March%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 2.5 Pro Preview (March 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    22.22%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $182.34
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922265_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 bg-blue-50/40">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">29</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                        <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Pareto optimal</span>
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20V3.1%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek V3.1 (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    20.00%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $12.55
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentdeepseekv31_1755793007_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">30</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20V3%20(March%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek V3 (March 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    17.78%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $25.26
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744854746_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">31</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o4-mini%20Low%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o4-mini Low (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    17.78%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $31.79
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745046580_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">32</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/o4-mini%20Low%20(April%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">o4-mini Low (April 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    15.56%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $22.50
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento4minilow_1755608756_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">33</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-OSS-120B%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-OSS-120B (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    11.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $4.21
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754492673_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">34</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-OSS-120B%20High%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-OSS-120B High (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    11.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $4.21
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754539776_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">35</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%202.0%20Flash%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 2.0 Flash (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    11.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $12.46
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744856042_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">36</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-5%20Medium%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-5 Medium (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    11.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $29.75
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgpt5medium_1756137340_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">37</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Claude%20Haiku%204.5%20(October%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Claude Haiku 4.5 (October 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    11.11%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $43.93
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudehaiku45_1760647306_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 bg-blue-50/40">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">38</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                        <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Pareto optimal</span>
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-OSS-120B%20High%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-OSS-120B High (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    8.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $2.05
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754569684_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">39</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/GPT-OSS-120B%20(August%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">GPT-OSS-120B (August 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    8.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $2.79
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754439280_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">40</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20V3%20(March%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek V3 (March 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    8.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $4.69
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentdeepseekv30324_1755710007_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">41</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20R1%20(May%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek R1 (May 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    8.89%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $7.77
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentdeepseekr10528_1755721934_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">42</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/CORE-Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            CORE-Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20R1%20(January%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek R1 (January 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    6.67%
                                    
                                        <span class="text-gray-500 ml-1">(-2.22/+2.22)</span>
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $81.11
                                
                                    <span class="text-gray-400 text-xs">(-46.45/+46.45)</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                2
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922373_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">43</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/DeepSeek%20R1%20(January%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">DeepSeek R1 (January 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    4.45%
                                    
                                        <span class="text-gray-500 ml-1">(-2.22/+2.22)</span>
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $24.95
                                
                                    <span class="text-gray-400 text-xs">(-11.07/+22.15)</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                2
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1747247500_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">44</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%202.0%20Flash%20(February%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 2.0 Flash (February 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    4.44%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $7.06
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgemini20flash_1755839828_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                        <tr class="hover:bg-gray-50 ">
                            <td class="px-6 py-4 whitespace-nowrap text-sm">45</td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <div class="text-sm">
                                    
                                        <a href="/agent/HAL%20Generalist%20Agent" class="text-blue-600 hover:text-blue-800 font-medium">
                                            HAL Generalist Agent
                                        </a>
                                    
                                    
                                </div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <a href="/model/Gemini%202.5%20Pro%20Preview%20(March%202025)" class="inline-block text-blue-600 hover:text-blue-800 mr-2 py-0.5 rounded font-medium">Gemini 2.5 Pro Preview (March 2025)</a>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                
                                    <span class="text-green-600">✓</span>
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                    4.44%
                                    
                                </span>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                $30.38
                                
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                1
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                
                                    <a href="https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1747247518_UPLOAD.zip?download=true" target="_blank" class="text-blue-600 hover:text-blue-800">
                                        Download
                                        <svg class="inline-block w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                                        </svg>
                                    </a>
                                
                            </td>
                        </tr>
                        
                    </tbody>
                </table>
            </div>
        
"""

soup = BeautifulSoup(html_content, 'html.parser')
table = soup.find('table')

# Extract headers
headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

# Extract rows
rows = []
for tr in table.find('tbody').find_all('tr'):
    cells = []
    for td in tr.find_all('td'):
        # For the "Verified" column, check for the checkmark
        if td.find('span', class_='text-green-600'):
            cells.append('Yes')
        # For the "Traces" column, extract the download link
        elif td.find('a', string='Download'):
            cells.append(td.find('a')['href'])
        else:
            # Clean up the text, preserving some structure for Rank/Model/Scaffold
            # Remove "Pareto optimal" badge text from Scaffold name for cleanliness, or keep it?
            # Let's keep the core text.
            text = td.get_text(separator=' ', strip=True)
            # Remove specific badge text if needed
            text = text.replace('Pareto optimal', '').strip()
            cells.append(text)
    rows.append(cells)

df = pd.DataFrame(rows, columns=headers)

# Save to CSV
df.to_csv('core.csv', index=False)

# Print head for verification
print(df.head())