using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Azure.CognitiveServices.Language.LUIS.Runtime;
using Microsoft.Azure.CognitiveServices.Language.TextAnalytics;
using Microsoft.Azure.CognitiveServices.Language.TextAnalytics.Models;

namespace MicrosoftTextAnalytics
{
    public class TextAnalyticsService
    {
        public TextAnalyticsService( string key, AzureRegions region )
        {
            var credentials = new ApiKeyServiceClientCredentials(key);
            _client = new Lazy<TextAnalyticsAPI>(() => new TextAnalyticsAPI(credentials));
            Client.AzureRegion = region;
        }

        public async Task<EntitiesBatchResult> GetEntities( List<MultiLanguageInput> documents, CancellationToken cancellationToken )
        {
            if (documents == null)
                return null;
            return await Client.EntitiesAsync(new MultiLanguageBatchInput(documents), cancellationToken).ConfigureAwait(false);
        }

        public async Task<KeyPhraseBatchResult> GetKeyPhrases( List<MultiLanguageInput> documents, CancellationToken cancellationToken )
        {
            if (documents == null)
                return null;
            return await Client.KeyPhrasesAsync(new MultiLanguageBatchInput(documents), cancellationToken).ConfigureAwait(false);
        }

        public async Task<SentimentBatchResult> GetSentiment( List<MultiLanguageInput> documents, CancellationToken cancellationToken )
        {
            if (documents == null)
                return null;
            return await Client.SentimentAsync(new MultiLanguageBatchInput(documents), cancellationToken).ConfigureAwait(false);
        }

        private TextAnalyticsAPI Client => _client.Value;
        private readonly Lazy<TextAnalyticsAPI> _client;
    }
}
