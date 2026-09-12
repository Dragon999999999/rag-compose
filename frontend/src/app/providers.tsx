import type { PropsWithChildren } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
const client = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 15_000 } } });
export function Providers({ children }: PropsWithChildren) { return <QueryClientProvider client={client}>{children}</QueryClientProvider>; }
