import React, { useState } from 'react';
import { Button } from './ui/button';
import { Textarea } from '../components/ui/textarea';
import { Card, CardContent } from '../components/ui/card';
import { useToast } from '../components/ui/use-toast';

interface AnalysisResult {
  isCyberbullying: boolean;
  confidence: number;
  category?: string;
}

const AnalysisForm = () => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const { toast } = useToast();

 //to be replaced with the AI API
  const analyzeCyberbullying = (text: string): AnalysisResult => {
    // Very basic detection logic for demonstration
    const bullying_phrases = [
      'hate you', 'stupid', 'ugly', 'kill yourself', 'loser', 
      'dumb', 'fat', 'worthless', 'pathetic', 'nobody likes you'
    ];
    
    const lowercaseText = text.toLowerCase();
    
    // Check if any bullying phrases exist in the text
    const foundPhrases = bullying_phrases.filter(phrase => 
      lowercaseText.includes(phrase)
    );
    
    const isCyberbullying = foundPhrases.length > 0;
    const confidence = isCyberbullying 
      ? Math.min(0.5 + (foundPhrases.length * 0.1), 0.95) 
      : 0.2;
      
    return {
      isCyberbullying,
      confidence,
      category: isCyberbullying ? 'Verbal Abuse' : undefined
    };
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!text.trim()) {
      toast({
        title: "Empty Text",
        description: "Please enter some text to analyze.",
        variant: "destructive",
      });
      return;
    }
    
    setIsAnalyzing(true);
    
    // Simulate API delay
    setTimeout(() => {
      const analysisResult = analyzeCyberbullying(text);
      setResult(analysisResult);
      setIsAnalyzing(false);
    }, 1000);
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-8">
      <form onSubmit={handleSubmit} className="space-y-4 ">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter or paste text to analyze for cyberbullying content..."
          className="min-h-[150px] border-cybersafe-200 focus-visible:ring-cybersafe-500"
        />
        
        <Button 
          type="submit" 
          disabled={isAnalyzing}
          className="w-auto mx-auto block bg-cybersafe-600 hover:bg-cybersafe-700 text-white "
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Text'}
        </Button>
      </form>

      {result && (
        <Card className="mt-8 overflow-hidden border-t-4 animate-in fade-in slide-in-from-top-5 duration-300" 
          style={{ 
            borderTopColor: result.isCyberbullying 
              ? '#ef4444'  // Red for cyberbullying 
              : '#10b981'  // Green for safe
          }}
        >
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                {result.isCyberbullying ? 'Potential Cyberbullying Detected' : 'No Cyberbullying Detected'}
              </h3>
              <div className="text-sm font-medium px-3 py-1 rounded-full" 
                style={{ 
                  backgroundColor: result.isCyberbullying 
                    ? '#fee2e2'  // Light red bg 
                    : '#d1fae5',  // Light green bg
                  color: result.isCyberbullying 
                    ? '#b91c1c'  // Dark red text 
                    : '#047857'  // Dark green text
                }}
              >
                {Math.round(result.confidence * 100)}% confidence
              </div>
            </div>
            
            <p className="text-gray-700">
              {result.isCyberbullying 
                ? 'This text contains language that may be considered harmful or offensive.' 
                : 'This text appears to be safe and does not contain obvious cyberbullying content.'}
            </p>
            
            {result.category && (
              <p className="mt-2 text-sm text-gray-500">
                Category: <span className="font-medium">{result.category}</span>
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AnalysisForm;
