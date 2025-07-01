// TraderAlpha - Asistente Global de IA
class TraderAlphaAssistant {
    constructor() {
        this.isInitialized = false;
        this.isLoading = false;
        this.conversationHistory = [];
        this.autoScrollEnabled = true;
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeElements();
            this.bindEvents();
            this.isInitialized = true;
            console.log('TraderAlpha Assistant initialized successfully');
        });
    }

    initializeElements() {
        this.fab = document.getElementById('traderalpha-fab');
        this.modal = document.getElementById('traderalpha-modal');
        this.closeButton = document.getElementById('traderalpha-close');
        this.form = document.getElementById('traderalpha-form');
        this.input = document.getElementById('traderalpha-input');
        this.messagesContainer = document.getElementById('traderalpha-messages');

        if (!this.fab || !this.modal) {
            console.warn('TraderAlpha elements not found in DOM');
            return;
        }
        
        // Configurar observador de mutaciones para scroll automático
        this.setupMutationObserver();
    }
    
    setupMutationObserver() {
        if (!this.messagesContainer) return;
        
        // Observar cambios en el contenido del chat
        this.mutationObserver = new MutationObserver((mutations) => {
            let shouldScroll = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldScroll = true;
                }
                if (mutation.type === 'characterData' || mutation.type === 'attributes') {
                    shouldScroll = true;
                }
            });
            
            if (shouldScroll) {
                this.forceScrollToBottom();
            }
        });
        
        // Iniciar observación
        this.mutationObserver.observe(this.messagesContainer, {
            childList: true,
            subtree: true,
            characterData: true,
            attributes: false
        });
    }

    bindEvents() {
        if (!this.fab || !this.modal) return;

        // Toggle modal
        this.fab.addEventListener('click', () => {
            this.toggleModal();
        });
        
        // Close button
        this.closeButton.addEventListener('click', () => {
            this.closeModal();
        });
        
        // Handle form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleUserMessage();
        });
        
        // Handle Enter key (send) and Shift+Enter (new line)
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleUserMessage();
            }
        });
        
        // Auto-resize textarea
        this.input.addEventListener('input', () => this.autoResizeTextarea());
        
        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            if (this.modal.classList.contains('hidden')) return;
            
            if (!this.modal.contains(e.target) && !this.fab.contains(e.target)) {
                this.closeModal();
            }
        });
    }

    toggleModal() {
        const isHidden = this.modal.classList.contains('hidden');
        if (isHidden) {
            this.openModal();
        } else {
            this.closeModal();
        }
    }

    openModal() {
        this.modal.classList.remove('hidden');
        
        // Add welcome message if this is the first interaction
        if (this.conversationHistory.length === 0) {
            this.addWelcomeMessage();
        }
        
        // Focus input and scroll to bottom with multiple attempts
        setTimeout(() => {
            this.input.focus();
            this.autoScrollEnabled = true;
            this.setupScrollDetection();
            this.forceScrollToBottom();
        }, 100);
        
        // Scroll adicional después de un delay más largo
        setTimeout(() => {
            this.forceScrollToBottom();
        }, 500);
    }

    closeModal() {
        this.modal.classList.add('hidden');
    }

    addWelcomeMessage() {
        const welcomeMessages = [
            "¡Hola! Soy TraderAlpha, tu asistente de análisis financiero. Puedo ayudarte con:",
            "• Análisis de derivados y opciones",
            "• Interpretación de datos de sentimiento",
            "• Explicación de conceptos de trading",
            "• Contexto de noticias macroeconómicas",
            "",
            "¿En qué puedo ayudarte hoy?"
        ];
        
        this.addMessage(welcomeMessages.join('\n'), 'ai');
    }

    handleUserMessage() {
        const userMessage = this.input.value.trim();
        if (!userMessage || this.isLoading) return;

        // Add user message to conversation
        this.addMessage(userMessage, 'user');
        this.conversationHistory.push({ role: 'user', content: userMessage });

        // Clear input
        this.input.value = '';
        this.autoResizeTextarea();

        // Fetch response from TraderAlpha
        this.fetchTraderAlphaResponse(userMessage);
    }

    addMessage(text, sender, isThinking = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender);
        
        const p = document.createElement('p');
        p.innerHTML = this.formatMessage(text);
        messageDiv.appendChild(p);

        if (isThinking) {
            messageDiv.classList.add('thinking');
        }

        this.messagesContainer.appendChild(messageDiv);
        
        // Hacer scroll inmediato y después de un delay
        this.scrollToBottom();
        setTimeout(() => {
            this.forceScrollToBottom();
        }, 50);
        
        return messageDiv;
    }

    formatMessage(text) {
        // Basic markdown-like formatting
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code style="background: rgba(142, 79, 255, 0.2); padding: 2px 4px; border-radius: 4px;">$1</code>');
    }

    autoResizeTextarea() {
        this.input.style.height = 'auto';
        this.input.style.height = Math.min(this.input.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        if (!this.messagesContainer) return;
        
        const container = this.messagesContainer;
        
        // Asegurar que el contenedor sea scrolleable
        this.ensureScrollable();
        
        // Función mejorada de scroll con múltiples intentos
        const doScroll = () => {
            const scrollHeight = container.scrollHeight;
            const clientHeight = container.clientHeight;
            const maxScroll = scrollHeight - clientHeight;
            
            if (maxScroll > 0) {
                // Método directo - forzar scroll al final
                container.scrollTop = scrollHeight + 1000; // Añadir extra por si acaso
                
                // ScrollIntoView como respaldo
                const lastMessage = container.lastElementChild;
                if (lastMessage) {
                    lastMessage.scrollIntoView({ 
                        behavior: 'instant', 
                        block: 'end',
                        inline: 'nearest'
                    });
                    
                    // Scroll adicional después de scrollIntoView
                    setTimeout(() => {
                        container.scrollTop = container.scrollHeight;
                    }, 10);
                }
            }
        };
        
        // Ejecutar inmediatamente
        doScroll();
        
        // Ejecutar después del render con múltiples intentos
        requestAnimationFrame(() => {
            doScroll();
            
            // Más intentos con diferentes delays
            setTimeout(doScroll, 50);
            setTimeout(doScroll, 100);
            setTimeout(doScroll, 200);
            
            // Intento final después de que el contenido se haya cargado completamente
            requestAnimationFrame(doScroll);
        });
        
        // Ejecutar después de delay para contenido dinámico
        setTimeout(doScroll, 50);
        setTimeout(doScroll, 150);
    }

    ensureScrollable() {
        if (!this.messagesContainer) return;
        
        // Asegurar que el contenedor tenga las propiedades correctas
        const container = this.messagesContainer;
        container.style.overflowY = 'auto';
        container.style.overflowX = 'hidden';
        container.style.scrollBehavior = 'auto';
        
        // Asegurar altura máxima para que sea scrolleable
        if (!container.style.maxHeight) {
            container.style.maxHeight = '400px';
        }
    }

    smoothScrollToBottom() {
        if (!this.messagesContainer) return;
        
        this.ensureScrollable();
        const container = this.messagesContainer;
        
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }

    forceScrollToBottom() {
        if (!this.messagesContainer) return;
        
        // Sistema de scroll definitivo
        const container = this.messagesContainer;
        this.ensureScrollable();
        
        let attempts = 0;
        const maxAttempts = 5;
        
        const forcedScroll = () => {
            const scrollHeight = container.scrollHeight;
            const clientHeight = container.clientHeight;
            
            // Scroll directo al final
            container.scrollTop = scrollHeight;
            
            // Verificar si llegó al final
            const currentScroll = container.scrollTop;
            const expectedScroll = scrollHeight - clientHeight;
            const isAtBottom = Math.abs(currentScroll - expectedScroll) <= 5;
            
            attempts++;
            
            if (!isAtBottom && attempts < maxAttempts) {
                requestAnimationFrame(forcedScroll);
            }
        };
        
        forcedScroll();
    }

    setupScrollDetection() {
        if (!this.messagesContainer) return;
        
        this.messagesContainer.addEventListener('scroll', () => {
            const container = this.messagesContainer;
            const isAtBottom = container.scrollTop >= container.scrollHeight - container.clientHeight - 50;
            
            // Si el usuario hace scroll hacia arriba, deshabilitar auto-scroll
            if (!isAtBottom) {
                this.autoScrollEnabled = false;
            } else {
                this.autoScrollEnabled = true;
            }
        });
    }


    async fetchTraderAlphaResponse(query) {
        this.isLoading = true;
        const thinkingMessage = this.addMessage('Analizando...', 'ai', true);

        try {
            const response = await fetch('/api/ask_traderalpha', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error en la respuesta del servidor.');
            }

            const data = await response.json();
            
            // Remove thinking message
            thinkingMessage.remove();
            
            // Add AI response with typing effect
            this.typeMessage(data.response, 'ai');
            
            // Add to conversation history
            this.conversationHistory.push({ role: 'assistant', content: data.response });

        } catch (error) {
            console.error('Error fetching TraderAlpha response:', error);
            
            // Remove thinking message and show error
            thinkingMessage.remove();
            
            const errorMessage = this.getErrorMessage(error);
            this.addMessage(errorMessage, 'ai');
            
        } finally {
            this.isLoading = false;
        }
    }

    getErrorMessage(error) {
        if (error.message.includes('configurada')) {
            return '🔧 Parece que hay un problema de configuración. El administrador debe configurar la clave API de TraderAlpha.';
        } else if (error.message.includes('fetch')) {
            return '🌐 Error de conexión. Por favor, verifica tu conexión a internet e inténtalo de nuevo.';
        } else {
            return `⚠️ Lo siento, he tenido un problema técnico: ${error.message}`;
        }
    }

    typeMessage(text, sender) {
        const messageDiv = this.addMessage('', sender);
        const p = messageDiv.querySelector('p');
        
        let currentText = '';
        let currentIndex = 0;
        let charCount = 0;
        
        const typeCharacter = () => {
            if (currentIndex < text.length) {
                currentText += text[currentIndex];
                p.innerHTML = this.formatMessage(currentText);
                currentIndex++;
                charCount++;
                
                // Scroll cada 5 caracteres para seguimiento constante
                if (charCount % 5 === 0 || text[currentIndex - 1] === '\n') {
                    this.forceScrollToBottom();
                }
                
                // Random delay between 20-50ms for more natural typing
                const delay = Math.random() * 30 + 20;
                setTimeout(typeCharacter, delay);
            } else {
                // Múltiples intentos de scroll al finalizar
                this.forceScrollToBottom();
                
                setTimeout(() => {
                    this.forceScrollToBottom();
                }, 100);
                
                setTimeout(() => {
                    this.forceScrollToBottom();
                }, 300);
                
                setTimeout(() => {
                    this.forceScrollToBottom();
                }, 600);
            }
        };
        
        typeCharacter();
    }

    // Public methods for external use
    openChat() {
        this.openModal();
    }

    askQuestion(question) {
        this.openModal();
        this.input.value = question;
        this.handleUserMessage();
    }

    clearConversation() {
        this.conversationHistory = [];
        this.messagesContainer.innerHTML = '';
        this.addWelcomeMessage();
    }
}

// Initialize TraderAlpha Assistant
const traderAlpha = new TraderAlphaAssistant();

// Export for external use
window.TraderAlpha = traderAlpha;
